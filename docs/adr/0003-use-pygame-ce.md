# 使用 pygame-ce 实现 Pygame 界面

图形界面的 optional extra 声明 `pygame-ce>=2.5.8,<3`，应用代码继续使用标准的 `pygame` import namespace，并且不同时安装或回退到上游 `pygame`。上游 pygame 2.6.1 没有覆盖项目开发环境 CPython 3.14 的 wheel，而 pygame-ce 覆盖项目声明的 Python 3.11–3.14，且所需基础 API 兼容；限制主版本可避免未来 pygame-ce 3 的变化被静默引入。
