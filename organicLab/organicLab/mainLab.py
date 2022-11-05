"""
Stubb of organic chemistry lab API. Project is private, message for details.

"""

from flask import Flask, Response, stream_with_context, send_from_directory, request
from flask_restful import Api, Resource, reqparse, fields, marshal_with, abort
import json, re
import chemFuncs
from utils import requestParser
from chemBlocks import getElement, Formula

app = Flask(__name__)
api = Api(app)
app.secret_key = 'our super secret lab key :)'

YOUR_DOMAIN = 'http://localhost:4242'

@app.before_request
def before_request_func():
    if request.url.startswith("http://127."):
        pass
    else:
        return None

ppmCalcArgs = requestParser()
ppmCalcArgs.add_argument("measured", type=float)
ppmCalcArgs.add_argument("theoretical", type=float)

class ppmCalc(Resource):
    def get(self):
        description = {
            "POST": {"Method": "POST",
             "Description": "Method to calculate the parts per million error of a measurement",
             "Arguments": {
                "measured": {
                    "Type": "Float",
                    "Description": "Measured mass of element or compound",
                    "Example": "12.002134"
                    },
                "theoretical" : {
                    "Type": "Float",
                    "Description": "Theoretical mass of element or compound",
                    "Example": "12.002134"
                    }
                }
             }
        }
        return json.dumps(description)

    def post(self):
        args = ppmCalcArgs.returnJSON()
        if (args['measured'] == None) or (args['theoretical'] == None):
            return "measured and theoretical masses must be floats"
        elif (args['measured'] < 0) or (args['theoretical'] < 0):
            return "measured and theoretical masses must be > 0"
        else:
            ppm = formulaFuncs.ppm_calc(args['measured'], args['theoretical'])
            return [*ppm]

api.add_resource(ppmCalc, "/ppmCalc")

calcMWArgs = requestParser()
calcMWArgs.add_argument("formula", type=str)

class calcMW(Resource):
    def get(self):
        description = {
            "POST": {"Method": "POST",
                     "Description": "Method to calculate the mass of a given empirical formula",
                     "Arguments": {
                         "formula": {
                             "Type": "string",
                             "Description": "a string empirical formula of a compound",
                             "Example": "C2H3O"
                         }
                     }
                     }
        }
        return json.dumps(description)

    def post(self):
        args = calcMWArgs.returnJSON()
        reggy = "^(([CHNOchno][0-9]{1,2}){1,4})$"
        if args['formula'] == None:
            return "No valid formula string submitted - use only C,N,O,H elements"
        elif bool(re.compile(reggy).match(args['formula'])) == False:
            return "Invalid formula string submitted - use only C,N,O,H elements"
        else:
            mw = formulaFuncs.calc_MW(args['formula'])
            return mw

api.add_resource(calcMW, "/calcMW")

getEleArgs = requestParser()
getEleArgs.add_argument("element", type=str)

class getElement(Resource):
    def get(self):
        description = {
            "POST": {"Method": "POST",
                     "Description": "Method to get Elemental Data",
                     "Arguments": {
                         "element": {
                             "Type": "string",
                             "Description": "symbol of element",
                             "Example": "K"
                         }
                     }
                     }
        }
        return json.dumps(description)

    def post(self):
        args = getEleArgs.returnJSON()
        if args['element'] == None:
            return "No valid Elemental string submitted"
        else:
            element = getElement(args['element'])
            if element == None:
                return "Element not found"
            else:
                return json.dumps(element)

api.add_resource(getElement, "/getElement")


spectrumPeakArgs = requestParser()
spectrumPeakArgs.add_argument("formula", type=str)

class calcMW(Resource):
    def get(self):
        description = {
            "POST": {"Method": "POST",
                     "Description": "Get estimated spectrum peak info for compound",
                     "Arguments": {
                         "formula": {
                             "Type": "string",
                             "Description": "a string formula of a compound",
                             "Example": "C2H3O"
                         }
                     }
                     }
        }
        return json.dumps(description)

    def post(self):
        args = spectrumPeakArgs.returnJSON()
        reggy = "^(([CHNOchno][0-9]{1,2}){1,4})$"
        if args['formula'] == None:
            return "No valid formula string submitted - use only C,N,O,H elements"
        elif bool(re.compile(reggy).match(args['formula'])) == False:
            return "Invalid formula string submitted - use only C,N,O,H elements"
        else:
            spectrum = Formula(args['formula'])
            return json.dumps(spectrum.peak.tolist())

api.add_resource(calcMW, "/calcMW")



if __name__ == '__main__':
    app.run()

