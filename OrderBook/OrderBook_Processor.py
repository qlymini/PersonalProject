import heapq
from util import dir_path
import queue
import argparse
from heap import orderQueue
import collections
from Order import Order
import csv
import time


class OrderBook:
    def __init__(self):
        self.asksqueue = orderQueue()
        self.bidsqueue = orderQueue()
        self.bids = []
        self.orders = collections.defaultdict(int)
        self.price_tracking = collections.defaultdict(tuple)
        self.orderbook = collections.defaultdict(list)
        self.ob_time = -1
        self.updates = queue.PriorityQueue()
        self.delete_queue = queue.PriorityQueue()

    def _handle_fourlargest(self, n, ablist):
        diff = n - len(ablist)
        ablist = list(sum(ablist, ()))
        ablist = ablist + [0] * diff * 2

        return ablist

    # functions to get order from updates priority queue and then updating orderbook by timestamp, price, side
    def _update_book(self, csv_file):
        if not self.updates.empty():
            info_tuple = self.updates.get()
            five_largest_ask = heapq.nlargest(5, self.asksqueue.get_queue(), lambda x: x[0])
            five_largest_bid = heapq.nlargest(5, self.bidsqueue.get_queue(), lambda x: x[0])

            five_asks = self._handle_fourlargest(5, five_largest_ask)
            five_bids = self._handle_fourlargest(5, five_largest_bid)

            timestampe, price, side, act = info_tuple

            if act == 'add' or act == 'update':
                info_list = [timestampe] + five_bids + five_asks
                self.orderbook[(price, side)] = info_list
                row = [timestampe, price, side] + info_list[1:]
                csv_file.writerow(row)
            if act == 'delete':
                row = [timestampe] + [0] * 10
                if (price, side) in self.orderbook:
                    self.orderbook[(price, side)] = row
                csv_file.writerow(row)

    # to get each order from csv and updating our orderqueue
    def get_order(self, order_list):
        timestamp = int(order_list[0])

        order = Order(side=order_list[1], action=order_list[2], id=int(order_list[3]),
                      price=float(order_list[4]), quantity=int(order_list[5]))

        if order.action == 'a':
            self.orders[(order.price, order.side)] = self.orders[(order.price, order.side)] + order.quantity
            self.price_tracking[order.id] = (order.price, order.side, order.quantity)
            self.updates.put((timestamp, order.price, order.side, 'add'))

            # updating order queue for action,it will add new order tuple to queue if we do not have this price, side level
            if order.side == 'a':
                self.asksqueue.update_task(new_quantity=order.quantity, priority=order.price)
            else:
                self.bidsqueue.update_task(new_quantity=order.quantity, priority=order.price)

        elif order.action == 'm':
            old_price, old_side, old_quantity = self.price_tracking[order.id]
            self.updates.put((timestamp, order.price, order.side, 'add'))

            # if we see order,side level have 0 quantity after modifying, then we delete it from our order dictionary
            # and also put delete info for this price side level into updates queue
            if self.orders[(old_price, old_side)] - old_quantity == 0:
                del self.orders[(old_price, old_side)]
                self.updates.put((timestamp, old_price, old_side, 'delete'))
            else:
                self.orders[(old_price, old_side)] = self.orders[(old_price, old_side)] - old_quantity
                self.updates.put((timestamp, old_price, old_side, 'update'))
            del self.price_tracking[order.id]
            self.price_tracking[order.id] = (order.price, order.side, order.quantity)
            self.orders[(order.price, order.side)] = self.orders[(order.price, order.side)] + order.quantity

            if order.side == 'a':
                self.asksqueue.update_task(new_quantity=old_quantity * (-1), priority=old_price)
                self.asksqueue.update_task(new_quantity=order.quantity, priority=order.price)
            else:
                self.bidsqueue.update_task(new_quantity=old_quantity * (-1), priority=old_price)
                self.bidsqueue.update_task(new_quantity=order.quantity, priority=order.price)

        # for delete action, remove all info of the orders in our system.
        elif order.action == 'd':
            old_price, old_side, old_quantity = self.price_tracking[order.id]

            if self.orders[(old_price, old_side)] - old_quantity == 0:
                del self.orders[(old_price, old_side)]
                self.updates.put((timestamp, old_price, old_side, 'delete'))
            else:
                self.orders[(old_price, old_side)] = self.orders[(old_price, old_side)] - old_quantity
                self.updates.put((timestamp, old_price, old_side, 'update'))

            del self.price_tracking[order.id]

            if order.side == 'a':
                self.asksqueue.update_task(new_quantity=old_quantity * (-1), priority=old_price)
            else:
                self.bidsqueue.update_task(new_quantity=old_quantity * (-1), priority=old_price)

    def book_updating_process(self, csvfile):
        self._update_book(csv_file=csvfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='order book arguement')
    parser.add_argument('--read', type=dir_path)
    parser.add_argument('--output', type=dir_path)
    args = parser.parse_args()
    reading_path = args.read
    output_path = args.output
    orderbook = OrderBook()

    with open(reading_path, "r") as f1:
        orders = f1.readlines()
    f = open(output_path, 'w', newline="")
    write = csv.writer(f)
    write.writerow(
        ['timestamp', 'price', 'side', 'bp0', 'bq0', 'bp1', 'bq1', 'bp2', 'bq2', 'bp3', 'bq3', 'bp4', 'bq4',
         'ap0', 'aq0', 'ap1', 'aq1', 'ap2', 'aq2', 'ap3', 'aq3', 'ap4', 'aq4', ])
    for i in orders[1:]:
        ls = i.rstrip().split(',')
        orderbook.get_order(ls)
        orderbook.book_updating_process(csvfile=write)
