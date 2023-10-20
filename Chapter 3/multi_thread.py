import threading
import time


class MyThread(threading.Thread):
    def __init__(self, name, delay):
        threading.Thread.__init__(self)
        self.name = name
        self.delay = delay


    def run(self):
        print('Starting thread %s.' % self.name)
        thread_lock.acquire()
        print_numbers(self.name, self.delay)
        thread_lock.release()


def print_numbers(threadName, delay):
    counter = 0
    while counter < 3:
        time.sleep(delay)
        print('%s: %s' % (threadName, time.ctime(time.time())))
        counter += 1


thread_lock = threading.Lock()
threads = []


# Create new threads
thread1 = MyThread("Thread-1", 1)
thread2 = MyThread("Thread-2", 2)


# Start new Threads
thread1.start()
thread2.start()


# Add threads to thread list
threads.append(thread1)
threads.append(thread2)


# Wait for all threads to complete
for t in threads:
    t.join()


print('Exiting Main Thread.')