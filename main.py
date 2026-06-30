from dotenv import load_dotenv
load_dotenv()
import torch
print(torch.cuda.is_available())       
print(torch.cuda.get_device_name(0))
if __name__ == "__main__":
    from llms import predict
    from agent import Agent
    ticker = "AAPL"
    year = 2018
    month = "May"
    day = 1
    agent = Agent()
    result = predict(agent, ticker, year=year, month=month, day=day)
    print(result)