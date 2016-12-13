'''
What does this module do?
1. Creates pubsub client and listen for messages.
2. Kicks off a client-defined long-running job.
3. Updates redis hash when job completes.
'''

from __future__ import print_function
import redis, sys

'''
Fetch job data and call processing function
'''
class Worker(object):
    def __init__(self, redis_store):
        self.send = redis_store

    '''
    run worker with given msg and args
    '''
    def run(self, msg, **kwargs):
        key = msg['data']
        initial_state = kwargs.get('initial_state', 'new')
        process_func = kwargs.get('process_func',
            lambda key, job: sys.stdout.write('processing placeholder'))

        # get object for corresponding key:
        job = self.send.hgetall(key)
        print(job)
        print(key)
        if not job:
            self.send.hmset(key, {'state': 'error', 'error': 'could not find item in redis'})
            print('COULD NOT FIND ITEM IN REDIS')
            return
        # for dupe publishings. job already running.
        if job['state'] != initial_state:
            print('NOT YET FINISHED')
            return

        job['state'] = 'processing'

        # reset results (if job was re-run)
        for k in ('error', 'data'):
            if k in job: job.pop(k)

        # update with new state
        self.send.hmset(key, job)

        try:
            process_func(key, job)
        except Exception as e:
            job['state'] = 'error'
            job['error'] = e
            print(e)
        # when done, update job
        self.send.hmset(key, job)

'''
Listen for messages and dispatch workers.
1 worker per message for parallelization support (TODO).
'''
class Dispatcher(object):
    '''
    Args:
        channels: list of channel names
        process_func: a blocking function with args: key (string), job (dict)
        initial_state: of redis job. ex. 'new'
    '''
    def __init__(self, redis_host='localhost', redis_port=6379, **kwargs):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.channels = kwargs['channels']
        self.process_func = kwargs['process_func']
        self.initial_state = kwargs.get('initial_state', 'new')

    def start(self):
        pool = redis.ConnectionPool(host=self.redis_host,
            port=self.redis_port, decode_responses=True)
        redis_store = redis.Redis(connection_pool=pool)
        try:
            # test connection
            redis_store.ping()
        except redis.exceptions.ConnectionError as rexc:
            print('''
            Cannot connect to host "{host}" port {port}.
            Perhaps add "{host}" to /etc/hosts in dev env.
            '''.format(host=self.redis_host, port=self.redis_port))
            print(rexc)
            return
        redis_subscriber = redis.Redis(connection_pool=pool)
        pubsub = redis_subscriber.pubsub()
        pubsub.subscribe(self.channels)
        # listen for messages and do work:
        for item in pubsub.listen():
            print('MESSAGE HEARD')
            if item['type'] == 'subscribe': # subscribe response
                print('SUBSCRIBED TO CHANNEL %s' % item['channel'])
            elif item['data'] == 'KILL':
                pubsub.unsubscribe()
                print('un-subscribed and finished')
                break
            else:
                worker = Worker(redis_store)
                try:
                    worker.run(item, process_func=self.process_func, initial_state=self.initial_state)
                except Exception as e:
                    print('error running redis worker:', e)

