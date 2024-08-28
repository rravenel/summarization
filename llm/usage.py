
# prices in, out
PRICE_GPT_4o_MINI = (0.00000015, 0.00000060)
PRICE_GPT_4o = (0.00000250, 0.00001000)

class Usage():
    #def __init__(self, price_in: float = PRICE_GPT_4o[0], price_out: float = PRICE_GPT_4o[1]) -> None:
    def __init__(self, price_in: float = PRICE_GPT_4o_MINI[0], price_out: float = PRICE_GPT_4o_MINI[1]) -> None:
        self.tokens_in = 0
        self.tokens_out = 0
        self.price_in = price_in
        self.price_out = price_out
        self.cost_total = 0.0

    def __str__(self) -> str:
        return f"Tokens in: {self.tokens_in}\tTokens out: {self.tokens_out}\tTotal cost: ${self.cost_total:.3f}"

    def calculate_cost(self) -> None:
        self.cost_total = self.tokens_in * self.price_in + self.tokens_out * self.price_out

    def add_usage(self, usage: 'Usage') -> None:
        self.tokens_in += usage.tokens_in
        self.tokens_out += usage.tokens_out
        self.cost_total += usage.cost_total

    def add_tokens(self, tokens_in: int, tokens_out: int) -> None:
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.calculate_cost()
