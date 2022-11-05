import json


class requestParser:
    def __init__(self):
        self.args = {}
        self.argTypes = {}

    def add_argument(self, name, type=str):
        self.args[name] = None
        self.argTypes[name] = type

    def returnArgs(self, req=None):
        if req == None:
            req = request
        tmpArgs = request.args
        args = {}
        for arg in tmpArgs:
            if arg not in self.args:
                pass
            else:
                if type(tmpArgs[arg]) != self.argTypes[arg]:
                    pass
                else:
                    args[arg] = tmpArgs[arg]
        for arg in self.args:
            if arg not in args:
                args[arg] = None
        return args

    def returnJSON(self, req=None, force=False):
        if req == None:
            req = request
        tmpArgs = json.loads(request.data)
        args = {}
        for arg in tmpArgs:
            if arg not in self.args:
                pass
            else:
                if force == False:
                    if type(tmpArgs[arg]) != self.argTypes[arg]:
                        pass
                    else:
                        args[arg] = tmpArgs[arg]
                else:
                    args[arg] = tmpArgs[arg]
        for arg in self.args:
            if arg not in args:
                args[arg] = None
        return args


