from multiprocessing import Process, Pipe, Queue
from multiprocessing.connection import Connection
import time

def f(conn: Connection):
    for i in range(1000):
        conn.send(["hello", i])
        print(f"sent {i}")
    

def g(q: Queue):
    for i in range(1000):
        q.put(["hello", i])
        print(f"sent {i}")


# p_conn, c_conn = Pipe()
q = Queue()

# p = Process(target=f, args=(c_conn,))
p = Process(target=g, args=(q,))
p.start()

for i in range(1000):
    # x = p_conn.recv()
    x = q.get()
    print(f"recv {x}")
    time.sleep(1)