<div align="center">
<h1 align="center">Search RAG</h1>
基于 <a href=https://github.com/leptonai/search_with_lepton>search_with_lepton</a> 修改的自部署版 RAG 应用
<br/>
<img width="70%" src="https://github.com/leptonai/search_with_lepton/assets/1506722/845d7057-02cd-404e-bbc7-60f4bae89680">
</div>


## 使用 docker 快速部署
1. 准备 bing search v7 的 API ，可自行 [google](https://google.com)，一般步骤为注册 Azure，搜索 bing search v7 开启 API，生成密钥复制过来即可
2. 准备 openai API,也可以使用 Gemini
3. 运行 `docker`，执行
    ```bash
    docker run -d -e BACKEND=bing -e BING_SEARCH_V7_SUBSCRIPTION_KEY=你的 bing 密钥 -e LLM_MODEL=gpt-3.5-turbo -e RELATED_QUESTIONS=0 -e OPENAI_API_KEY=你的 openai 密钥 -e OPENAI_BASE_URL=https://api.openai.com/v1/ ccoder64/search_rag:latest
    ```
4. 访问 http://localhost:8080

使用 Gemini 步骤：
- 去 https://ai.google.dev/ 获取 API KEY
- 可自行搭建 gemini-openai-proxy，也可使用 `https://gemini-pro-openai-proxy.deno.dev/v1/`
- 覆盖环境变量 OPENAI_API_KEY=你的Gemini 密钥，OPENAI_BASE_URL=搭建好的 proxy 地址
