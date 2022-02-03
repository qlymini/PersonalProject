import heapq
from heapq import heappush, heappop
from dataclasses import dataclass, field


@dataclass(order=True)
class orderQueue:
    priority: int
    quantity: int = field(compare=False)

    def __init__(self):
        self.pq = []
        self.entry_finder = {}  # entry searching dictionary for helping us finding and update order in heapq

    def add_task(self, quantity, priority=0):
        self.entry_finder[priority] = quantity
        heappush(self.pq, (priority, quantity))

    # it will add new task if we cannot not find the priority in our entry searching dictionary
    def update_task(self, new_quantity, priority):
        if priority in self.entry_finder:
            old_quant = self.entry_finder[priority]
            self.remove_task(priority)
            new = old_quant + new_quantity
            if new > 0:
                self.entry_finder[priority] = new
                heappush(self.pq, (priority, new))
        else:
            self.add_task(quantity=new_quantity, priority=priority)

    def remove_task(self, priority):
        quantity = self.entry_finder.pop(priority)
        self.pq.remove((priority, quantity))
        heapq.heapify(self.pq)

    def pop_task(self):
        if self.pq:
            heappop(self.pq)
        raise KeyError('pop from an empty priority queue')

    def get_queue(self):
        return self.pq


if __name__ == '__main__':
    q = orderQueue()
    q.add_task(quantity=12, priority=22)
    q.add_task(quantity=10, priority=2)
    q.add_task(quantity=12, priority=20)
    dq = q.get_queue()
    print(dq)
    q.update_task(new_quantity=-90, priority=4)
    dq = q.get_queue()
    print(dq)
    q.remove_task(priority=22)
    dq = q.get_queue()
    print(dq)
    print(heapq.nlargest(3, dq, key=lambda x: x[1]))
