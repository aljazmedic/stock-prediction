

class FundNameException(BaseException): pass


class FundOutFileNameException(BaseException): pass


class FundNotFoundException(BaseException): pass


class StockInformationMissingException(BaseException): pass


class StockCandleInformatinMissingException(StockInformationMissingException): pass