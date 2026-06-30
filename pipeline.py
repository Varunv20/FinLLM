class Pipeline():
    def __init__(self, ticker):
        self.ticker = ticker

    def run(self, data):
        for step in self.steps:
            data = step.process(data)
        return data