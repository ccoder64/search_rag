import concurrent.futures
import json
import os
import re
import requests
import traceback
import httpx
from typing import Annotated, List, Generator
from openai import AsyncOpenAI
from loguru import logger

import sanic
from sanic import Sanic
import sanic.exceptions
from sanic.exceptions import HTTPException, InvalidUsage
from sqlitedict import SqliteDict

app = Sanic("search")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


################################################################################
# Constant values for the RAG model.
################################################################################

# Search engine related. You don't really need to change this.
BING_SEARCH_V7_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"
BING_MKT = "en-US"
GOOGLE_SEARCH_ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"
SERPER_SEARCH_ENDPOINT = "https://google.serper.dev/search"
SEARCHAPI_SEARCH_ENDPOINT = "https://www.searchapi.io/api/v1/search"


# Specify the number of references from the search engine you want to use.
# 8 is usually a good number.
REFERENCE_COUNT = 8

# Specify the default timeout for the search engine. If the search engine
# does not respond within this time, we will return an error.
DEFAULT_SEARCH_ENGINE_TIMEOUT = 5


# If the user did not provide a query, we will use this default query.
_default_query = "Who said 'live long and prosper'?"

# This is really the most important part of the rag model. It gives instructions
# to the model on how to generate the answer. Of course, different models may
# behave differently, and we haven't tuned the prompt to make it optimal - this
# is left to you, application creators, as an open problem.
_rag_query_text = """
You are a large language AI assistant built by Lepton AI. You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable.

Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

Please cite the contexts with the reference numbers, in the format [citation:x]. If a sentence comes from multiple contexts, please list all applicable citations, like [citation:3][citation:5]. Other than code and specific names and citations, your answer must be written in the same language as the question.

Here are the set of contexts:

{context}

Remember, don't blindly repeat the contexts verbatim. And here is the user question:
"""

# A set of stop words to use - this is not a complete set, and you may want to
# add more given your observation.
stop_words = [
    "<|im_end|>",
    "[End]",
    "[end]",
    "\nReferences:\n",
    "\nSources:\n",
    "End.",
]

# This is the prompt that asks the model to generate related questions to the
# original question and the contexts.
# Ideally, one want to include both the original question and the answer from the
# model, but we are not doing that here: if we need to wait for the answer, then
# the generation of the related questions will usually have to start only after
# the whole answer is generated. This creates a noticeable delay in the response
# time. As a result, and as you will see in the code, we will be sending out two
# consecutive requests to the model: one for the answer, and one for the related
# questions. This is not ideal, but it is a good tradeoff between response time
# and quality.
_more_questions_prompt = """
You are a helpful assistant that helps the user to ask related questions, based on user's original question and the related contexts. Please identify worthwhile topics that can be follow-ups, and write questions no longer than 20 words each. Please make sure that specifics, like events, names, locations, are included in follow up questions so they can be asked standalone. For example, if the original question asks about "the Manhattan project", in the follow up question, do not just say "the project", but use the full name "the Manhattan project". Your related questions must be in the same language as the original question.

Here are the contexts of the question:

{context}

Remember, based on the original question and related contexts, suggest three such further questions. Do NOT repeat the original question. Each related question should be no longer than 20 words. Here is the original question:
"""


class KVWrapper(object):
    def __init__(self, kv_name):
        self._db = SqliteDict(filename=kv_name)

    def get(self, key: str):
        v = self._db[key]
        if v is None:
            raise KeyError(key)
        return v

    def put(self, key: str, value: str):
        self._db[key] = value
        self._db.commit()


def search_with_bing(query: str, subscription_key: str):
    """
    Search with bing and return the contexts.
    """
    params = {"q": query, "mkt": BING_MKT}
    response = requests.get(
        BING_SEARCH_V7_ENDPOINT,
        headers={"Ocp-Apim-Subscription-Key": subscription_key},
        params=params,
        timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT,
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException("Search engine error.")
    json_content = response.json()
    try:
        contexts = json_content["webPages"]["value"][:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []
    return contexts


def search_with_google(query: str, subscription_key: str, cx: str):
    """
    Search with google and return the contexts.
    """
    params = {
        "key": subscription_key,
        "cx": cx,
        "q": query,
        "num": REFERENCE_COUNT,
    }
    response = requests.get(
        GOOGLE_SEARCH_ENDPOINT, params=params, timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException("Search engine error.")
    json_content = response.json()
    try:
        contexts = json_content["items"][:REFERENCE_COUNT]
        for item in contexts:
            item["name"] = item["title"]
            item["url"] = item["link"]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []
    return contexts


def search_with_serper(query: str, subscription_key: str):
    """
    Search with serper and return the contexts.
    """
    payload = json.dumps(
        {
            "q": query,
            "num": (
                REFERENCE_COUNT
                if REFERENCE_COUNT % 10 == 0
                else (REFERENCE_COUNT // 10 + 1) * 10
            ),
        }
    )
    headers = {"X-API-KEY": subscription_key, "Content-Type": "application/json"}
    logger.info(
        f"{payload} {headers} {subscription_key} {query} {SERPER_SEARCH_ENDPOINT}"
    )
    response = requests.post(
        SERPER_SEARCH_ENDPOINT,
        headers=headers,
        data=payload,
        timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT,
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException("Search engine error.")
    json_content = response.json()
    try:
        # convert to the same format as bing/google
        contexts = []
        if json_content.get("knowledgeGraph"):
            url = json_content["knowledgeGraph"].get("descriptionUrl") or json_content[
                "knowledgeGraph"
            ].get("website")
            snippet = json_content["knowledgeGraph"].get("description")
            if url and snippet:
                contexts.append(
                    {
                        "name": json_content["knowledgeGraph"].get("title", ""),
                        "url": url,
                        "snippet": snippet,
                    }
                )
        if json_content.get("answerBox"):
            url = json_content["answerBox"].get("url")
            snippet = json_content["answerBox"].get("snippet") or json_content[
                "answerBox"
            ].get("answer")
            if url and snippet:
                contexts.append(
                    {
                        "name": json_content["answerBox"].get("title", ""),
                        "url": url,
                        "snippet": snippet,
                    }
                )
        contexts += [
            {"name": c["title"], "url": c["link"], "snippet": c.get("snippet", "")}
            for c in json_content["organic"]
        ]
        return contexts[:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []


def search_with_searchapi(query: str, subscription_key: str):
    """
    Search with SearchApi.io and return the contexts.
    """
    payload = {
        "q": query,
        "engine": "google",
        "num": (
            REFERENCE_COUNT
            if REFERENCE_COUNT % 10 == 0
            else (REFERENCE_COUNT // 10 + 1) * 10
        ),
    }
    headers = {
        "Authorization": f"Bearer {subscription_key}",
        "Content-Type": "application/json",
    }
    logger.info(
        f"{payload} {headers} {subscription_key} {query} {SEARCHAPI_SEARCH_ENDPOINT}"
    )
    response = requests.get(
        SEARCHAPI_SEARCH_ENDPOINT,
        headers=headers,
        params=payload,
        timeout=30,
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise HTTPException("Search engine error.")
    json_content = response.json()
    try:
        # convert to the same format as bing/google
        contexts = []

        if json_content.get("answer_box"):
            if json_content["answer_box"].get("organic_result"):
                title = (
                    json_content["answer_box"].get("organic_result").get("title", "")
                )
                url = json_content["answer_box"].get("organic_result").get("link", "")
            if json_content["answer_box"].get("type") == "population_graph":
                title = json_content["answer_box"].get("place", "")
                url = json_content["answer_box"].get("explore_more_link", "")

            title = json_content["answer_box"].get("title", "")
            url = json_content["answer_box"].get("link")
            snippet = json_content["answer_box"].get("answer") or json_content[
                "answer_box"
            ].get("snippet")

            if url and snippet:
                contexts.append({"name": title, "url": url, "snippet": snippet})

        if json_content.get("knowledge_graph"):
            if json_content["knowledge_graph"].get("source"):
                url = json_content["knowledge_graph"].get("source").get("link", "")

            url = json_content["knowledge_graph"].get("website", "")
            snippet = json_content["knowledge_graph"].get("description")

            if url and snippet:
                contexts.append(
                    {
                        "name": json_content["knowledge_graph"].get("title", ""),
                        "url": url,
                        "snippet": snippet,
                    }
                )

        contexts += [
            {"name": c["title"], "url": c["link"], "snippet": c.get("snippet", "")}
            for c in json_content["organic_results"]
        ]

        if json_content.get("related_questions"):
            for question in json_content["related_questions"]:
                if question.get("source"):
                    url = question.get("source").get("link", "")
                else:
                    url = ""

                snippet = question.get("answer", "")

                if url and snippet:
                    contexts.append(
                        {
                            "name": question.get("question", ""),
                            "url": url,
                            "snippet": snippet,
                        }
                    )

        return contexts[:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []


def new_async_client(_app):
    return AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        http_client=_app.ctx.http_session,
    )


@app.before_server_start
async def server_init(_app, loop):
    """
    Initializes global configs.
    """
    _app.ctx.backend = os.environ["BACKEND"].upper()
    # if _app.ctx.backend == "LEPTON":
    #     from leptonai import Client

    #     _app.ctx.leptonsearch_client = Client(
    #         "https://search-api.lepton.run/",
    #         token=os.environ.get("LEPTON_WORKSPACE_TOKEN"),
    #         stream=True,
    #         timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10),
    #     )
    if _app.ctx.backend == "BING":
        _app.ctx.search_api_key = os.environ["BING_SEARCH_V7_SUBSCRIPTION_KEY"]
        _app.ctx.search_function = lambda query: search_with_bing(
            query,
            _app.ctx.search_api_key,
        )
    elif _app.ctx.backend == "GOOGLE":
        _app.ctx.search_api_key = os.environ["GOOGLE_SEARCH_API_KEY"]
        _app.ctx.search_function = lambda query: search_with_google(
            query,
            _app.ctx.search_api_key,
            os.environ["GOOGLE_SEARCH_CX"],
        )
    elif _app.ctx.backend == "SERPER":
        _app.ctx.search_api_key = os.environ["SERPER_SEARCH_API_KEY"]
        _app.ctx.search_function = lambda query: search_with_serper(
            query,
            _app.ctx.search_api_key,
        )
    elif _app.ctx.backend == "SEARCHAPI":
        _app.ctx.search_api_key = os.environ["SEARCHAPI_API_KEY"]
        _app.ctx.search_function = lambda query: search_with_searchapi(
            query,
            _app.ctx.search_api_key,
        )
    else:
        raise RuntimeError("Backend must be BING, GOOGLE, SERPER or SEARCHAPI.")
    _app.ctx.model = os.environ["LLM_MODEL"]
    _app.ctx.handler_max_concurrency = 16
    # An executor to carry out async tasks, such as uploading to KV.
    _app.ctx.executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=_app.ctx.handler_max_concurrency * 2
    )
    # Create the KV to store the search results.
    logger.info("Creating KV. May take a while for the first time.")
    _app.ctx.kv = KVWrapper(os.getenv("KV_NAME") or "search.db")
    # whether we should generate related questions.
    _app.ctx.should_do_related_questions = bool(
        os.environ["RELATED_QUESTIONS"] in ("1", "yes", "true")
    )
    # Create httpx Session
    _app.ctx.http_session = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10),
    )


async def get_related_questions(_app, query, contexts):
    """
    Gets related questions based on the query and context.
    """

    def ask_related_questions(
        questions: Annotated[
            List[str],
            [
                (
                    "question",
                    Annotated[
                        str,
                        "related question to the original question and context.",
                    ],
                )
            ],
        ],
    ):
        """
        ask further questions that are related to the input and output.
        """
        pass

    try:
        openai_client = new_async_client(_app)
        response = await openai_client.chat.completions.create(
            model=_app.ctx.model,
            messages=[
                {
                    "role": "system",
                    "content": _more_questions_prompt.format(
                        context="\n\n".join([c["snippet"] for c in contexts])
                    ),
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            tools=[
                {
                    "type": "function",
                    # "function": tool.get_tools_spec(ask_related_questions),
                }
            ],
            max_tokens=512,
        )
        related = response.choices[0].message.tool_calls[0].function.arguments
        if isinstance(related, str):
            related = json.loads(related)
        logger.trace(f"Related questions: {related}")
        return related["questions"][:5]
    except Exception as e:
        # For any exceptions, we will just return an empty list.
        logger.error(
            "encountered error while generating related questions:"
            f" {e}\n{traceback.format_exc()}"
        )
        return []


async def _raw_stream_response(
    _app, contexts, llm_response, related_questions_future
) -> Generator[str, None, None]:
    """
    A generator that yields the raw stream response. You do not need to call
    this directly. Instead, use the stream_and_upload_to_kv which will also
    upload the response to KV.
    """
    # First, yield the contexts.
    yield json.dumps(contexts)
    yield "\n\n__LLM_RESPONSE__\n\n"
    # Second, yield the llm response.
    if not contexts:
        # Prepend a warning to the user
        yield (
            "(The search engine returned nothing for this query. Please take the"
            " answer with a grain of salt.)\n\n"
        )
    async for chunk in llm_response:
        if chunk.choices:
            yield chunk.choices[0].delta.content or ""
    # Third, yield the related questions. If any error happens, we will just
    # return an empty list.
    if related_questions_future is not None:
        related_questions = await related_questions_future
        try:
            result = json.dumps(related_questions)
        except Exception as e:
            logger.error(f"encountered error: {e}\n{traceback.format_exc()}")
            result = "[]"
        yield "\n\n__RELATED_QUESTIONS__\n\n"
        yield result


def get_query_object(request):
    params = {k: v[0] for k, v in request.args.items()}
    if request.method == "POST":
        if "form" in request.content_type:
            params.update({k: v[0] for k, v in request.form.items()})
        else:
            try:
                if request.json:
                    params.update(request.json)
            except InvalidUsage:
                pass
    return params


@app.route("/query", methods=["POST"])
async def query_function(request: sanic.Request):
    """
    Query the search engine and returns the response.

    The query can have the following fields:
        - query: the user query.
        - search_uuid: a uuid that is used to store or retrieve the search result. If
            the uuid does not exist, generate and write to the kv. If the kv
            fails, we generate regardless, in favor of availability. If the uuid
            exists, return the stored result.
        - generate_related_questions: if set to false, will not generate related
            questions. Otherwise, will depend on the environment variable
            RELATED_QUESTIONS. Default: true.
    """
    _app = request.app
    params = get_query_object(request)
    query = params.get("query", None)
    search_uuid = params.get("search_uuid", None)
    generate_related_questions = params.get("generate_related_questions", True)
    if not query:
        raise HTTPException("query must be provided.")
    # Note that, if uuid exists, we don't check if the stored query is the same
    # as the current query, and simply return the stored result. This is to enable
    # the user to share a searched link to others and have others see the same result.
    if search_uuid:
        try:
            result = await _app.loop.run_in_executor(
                _app.ctx.executor, lambda sid: _app.ctx.kv.get(sid), search_uuid
            )
            return sanic.text(result)
        except KeyError:
            logger.info(f"Key {search_uuid} not found, will generate again.")
        except Exception as e:
            logger.error(
                f"KV error: {e}\n{traceback.format_exc()}, will generate again."
            )
    else:
        raise HTTPException("search_uuid must be provided.")

    # if _app.ctx.backend == "LEPTON":
    #     # delegate to the lepton search api.
    #     result = _app.ctx.leptonsearch_client.query(
    #         query=query,
    #         search_uuid=search_uuid,
    #         generate_related_questions=generate_related_questions,
    #     )
    #     return StreamingResponse(content=result, media_type="text/html")

    # First, do a search query.
    # query = query or _default_query
    # Basic attack protection: remove "[INST]" or "[/INST]" from the query
    query = re.sub(r"\[/?INST\]", "", query)
    contexts = await _app.loop.run_in_executor(
        _app.ctx.executor, _app.ctx.search_function, query
    )

    system_prompt = _rag_query_text.format(
        context="\n\n".join(
            [f"[[citation:{i+1}]] {c['snippet']}" for i, c in enumerate(contexts)]
        )
    )
    try:
        openai_client = new_async_client(_app)
        llm_response = await openai_client.chat.completions.create(
            model=_app.ctx.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            max_tokens=1024,
            stop=stop_words,
            stream=True,
            temperature=0.9,
        )
        if _app.ctx.should_do_related_questions and generate_related_questions:
            # While the answer is being generated, we can start generating
            # related questions as a future.
            related_questions_future = get_related_questions(_app, query, contexts)

        else:
            related_questions_future = None
    except Exception as e:
        logger.error(f"encountered error: {e}\n{traceback.format_exc()}")
        return sanic.json({"message": "Internal server error."}, 503)

    response = await request.respond(content_type="text/html")
    # First, stream and yield the results.
    all_yielded_results = []

    async for result in _raw_stream_response(
        _app, contexts, llm_response, related_questions_future
    ):
        all_yielded_results.append(result)
        await response.send(result)

    # Second, upload to KV. Note that if uploading to KV fails, we will silently
    # ignore it, because we don't want to affect the user experience.
    await response.eof()
    _ = _app.ctx.executor.submit(
        _app.ctx.kv.put, search_uuid, "".join(all_yielded_results)
    )


app.static("/ui", os.path.join(BASE_DIR, "ui/"), name="/")
app.static("/", os.path.join(BASE_DIR, "ui/index.html"), name="ui")


if __name__ == "__main__":
    port = int(os.getenv("PORT") or 8800)
    workers = int(os.getenv("WORKERS") or 1)
    app.run(host="0.0.0.0", port=port, workers=workers, debug=False)
