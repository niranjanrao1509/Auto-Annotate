import os
import json
import base64
#import jsbeautifier
#opts = jsbeautifier.default_options()
#opts.indent_size = 2
from misc import *
from flask import Flask
from flask import url_for
import base64
from flask_json import FlaskJSON, JsonError, json_response, as_json
from celery import Celery
from celery.result import AsyncResult
import celery.states as states

env = os.environ
CELERY_BROKER_URL = env.get('CELERY_BROKER_URL', 'pyamqp://' + RBMQ_USERNAME +
                            ':' + RBMQ_PASSWORD + '@' + RBMQ_HOST_IPADDR + ':' + str(RBMQ_PORT) + '/' + RBMQ_HOSTNAME),
CELERY_RESULT_BACKEND = env.get('CELERY_RESULT_BACKEND', 'amqp')
celery = Celery('tp_api',
                broker=CELERY_BROKER_URL,
                backend=CELERY_RESULT_BACKEND)

env = os.environ
app = Flask(__name__)




@app.route('/classify/<param1>/<param2>/<int:param3>/<int:param4>/<param5>')
def classify(param1, param2, param3, param4,param5):
    print('helllllllloooooo')
    print(param1)
    print(param2)
    print(param4)
    path=param1
    #path = base64.b64decode(param1)
    #path= path.decode('utf-8')
    print(path)
    model_name = base64.b64decode(param2)
    model_name = model_name[:-2]
    model_name=model_name.decode('utf-8')
    if(param5=='1'):
        print('param5 is 1')
        task = celery.send_task('tikapam.classify', args=[path, model_name, param3, param4])
        results = task.get()
        results = json.dumps(results)
        return (results)

    elif(param5=='2'):
        if(param4==4):
            task = celery.send_task('tikapam.classify_text', args=[path, model_name, param4])
            results = task.get()
            results = json.dumps(results)
            return (results)
        elif(param4==5):
            print('yoooooooo')
            task = celery.send_task('tikapam.senti_text', args=[path, model_name, param4])
            results = task.get()
            results = json.dumps(results)
            return (results)

        elif(param4==6):
            task = celery.send_task('tikapam.multi_text', args=[path, model_name, param4])
            results = task.get()
            results = json.dumps(results)
            return (results)

    else:
            #path = base64.b64decode(param1)
            #path= path.decode('utf-8')
            task = celery.send_task('tikapam.classify_audio', args=[path, model_name, param4])
            results = task.get()
            results = json.dumps(results)
            return (results)
    

@app.route('/landmrk/<param1>/<param2>/<int:param3>/<int:param4>')
def landmrk(param1, param2, param3, param4):
    print('flasssskkk')
    path=param1
    #path = base64.b64decode(param1)
    #path= path.decode('utf-8')
    model_name = base64.b64decode(param2)
    model_name = model_name[:-2]
    model_name=model_name.decode('utf-8')
    print('hereeee?')
    task = celery.send_task('tikapam.landmrk', args=[path, model_name, param3, param4])
    print('nowww')
    results = task.get()
    #with open('lnd.txt', 'w') as outfile:
    #    json.dump(results, outfile)
    return results



@app.route('/bbox/<param1>/<param2>/<int:param3>/<int:param4>')
def bbox(param1,param2,param3,param4):
    print('requested flask')
    path=param1
    #path = base64.b64decode(param1)
    #path= path.decode('utf-8')
    model_name = base64.b64decode(param2)
    model_name = model_name[:-2]
    model_name=model_name.decode('utf-8')
    print('sending task to celery')
    task = celery.send_task('tikapam.bbox', args=[path, model_name, param3, param4])
    results = task.get()
    print(results)
    #results = json.dumps(results)
    return (results)
#    with open('x.json', 'w') as outfile:
#        results=json.dumps(results)
#    print(results)
#    return (results.decode('utf-8'))


@app.route('/sseg/<param1>/<int:param2>')
def sseg(param1, param2):
    print('requested flask')
    path=param1
    print(param2)
    #path = base64.b64decode(param1)
    #path= path.decode('utf-8')
    print('sending task to celery')
    task = celery.send_task('tikapam.sseg', args=[path, param2])
    print('task done')
    results= str(task.get())
    with open('ss.json', 'w') as outfile:
        json.dump(results, outfile)
    return str(task.get())


@app.route('/check/<string:id>')
def check_task(id):
    res = celery.AsyncResult(id)
    return res.state if res.state == states.PENDING else str(res.result)


if __name__ == '__main__':
    app.run(debug=env.get('DEBUG', True),
            port=int(env.get('PORT', TP_API_ENGINE_PORT)),
            host=env.get('HOST', '127.0.0.1'))
