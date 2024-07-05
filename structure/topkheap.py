import heapq


class TopKHeap:
    def __init__(self, top_k):
        self.queue_ = []
        self.top_k = top_k
        self.index_ = 0

    def push(self, score, item):
        heapq.heappush(self.queue_, (score, self.index_, item))
        self.index_ += 1
        if len(self.queue_) == self.top_k + 1:
            heapq.heappop(self.queue_)

    def result(self):
        return self.queue_
