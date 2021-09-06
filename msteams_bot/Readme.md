# Teamsボット #

## 作成 ##

[Bot Framework SDK for Python を使用したボットの作成](https://docs.microsoft.com/ja-jp/azure/bot-service/python/bot-builder-python-quickstart?view=azure-bot-service-4.0)

[botbuilder-python](https://github.com/microsoft/botbuilder-python/#packages)
[BotFramework-Emulator](https://github.com/Microsoft/BotFramework-Emulator/blob/master/README.md)

[teams-conversation-bot](https://github.com/microsoft/BotBuilder-Samples/tree/main/samples/python/57.teams-conversation-bot)

~~~cmd
pip install botbuilder-core
pip install asyncio
pip install aiohttp
pip install cookiecutter==1.7.0

cd <dir>
cookiecutter https://github.com/microsoft/BotBuilder-Samples/releases/download/Templates/echo.zip

cd <bot-dir>
pip install -r requirements.txt
~~~