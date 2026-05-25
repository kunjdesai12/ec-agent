from rq import Worker, Queue, Connection
from redis import Redis

redis_conn = Redis(host="localhost", port=6379, db=0)

if __name__ == "__main__":
    with Connection(redis_conn):
        worker = Worker(queues=["stt_queue"])
        worker.work()