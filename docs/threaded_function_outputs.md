# How to get function outputs from another thread?

there are several other ways to run a function in a Thread and get its output in Python. One common alternative is to use the [threading.Thread](vscode-file://vscode-app/c:/Users/Saeed/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) subclass with a custom class. Here's an example of how we could rewrite this part using different approaches:

1. **Wrap in a function**

   ```python
   def threaded_consumer(q, n):
       out = []
       for _ in range(n):
           item = q.get()
           out.append(item)
       return out
   
   results = []
   def consumer_wrapper():
       results.extend(threaded_consumer(q, n))
   
   consumer = threading.Thread(target=consumer_wrapper)
   consumer.start()
   consumer.join()
   ```

2. **Using a Thread Subclass (Common Pattern)**:

   ```python
   class ThreadWithResult(threading.Thread):
       def __init__(self, target, args=()):
           super().__init__()
           self.target = target
           self.args = args
           self.result = None
   
       def run(self):
           self.result = self.target(*self.args)
   
   # Usage would be:
   consumer = ThreadWithResult(target=threaded_consumer, args=(q, n))
   consumer.start()
   consumer.join()
   result = consumer.result  # Get the result directly
   ```

3. **Using concurrent.futures (Modern Approach)**:

   ```python
   from concurrent.futures import ThreadPoolExecutor
   
   # Usage would be:
   with ThreadPoolExecutor(max_workers=2) as executor:
       future = executor.submit(threaded_consumer, q, n)
       result = future.result()  # This blocks until the result is ready
   ```

   max_workers:

   ​	Specifies the maximum number of threads that can be active simultaneously in the thread pool

   ​	The default is `min(32, os.cpu_count() + 4)`, which is more than we need

4. **Using Queue to Return Results (Another Common Pattern)**:

   ```python
   result_queue = queue.Queue()
   def thread_func():
       result = threaded_consumer(q, n)
       result_queue.put(result)
   
   consumer = threading.Thread(target=thread_func)
   consumer.start()
   consumer.join()
   result = result_queue.get()
   ```

Each approach has its advantages:

- `ThreadWithResult`: Good for object-oriented code and when you need more control over the thread
- `ThreadPoolExecutor`: Clean, modern, and handles thread lifecycle management
- Queue approach: Good for more complex scenarios where you need to pass multiple results or stream data