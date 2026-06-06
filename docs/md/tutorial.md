教程

环境变量怎么填？

如果使用官方的 api, 不需要自定义 baseurl
model 需要 provider/model 格式
```
COORDINATOR_API_KEY=xxx
COORDINATOR_MODEL=gemini/gemini-2.5-flash-preview-04-17
```

我使用中转该如何填写？

填写 baseurl
模型也需要 provider / model 格式

```
COORDINATOR_API_KEY=xxxx
COORDINATOR_MODEL=openai/DeepSeek-V3-Fast
COORDINATOR_BASE_URL=xxx
```

为什么使用 litellm 
因为各个厂家 api 格式各不相同
litellm 方便开发者将所有模型格式全都集成


