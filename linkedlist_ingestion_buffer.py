import time

class PacketNode:
    def __init__(self, generated_at: float, data: dict):
        self.generated_at = generated_at
        self.data = data
        self.prev = None
        self.next = None

class TimeSeriesLinkedList:
    """
    A Doubly Linked List optimized for time-series ingestion.
    Since out-of-order BLE packets usually arrive only slightly late,
    inserting by traversing backward from the tail is nearly O(1).
    """
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def insert(self, generated_at: float, data: dict):
        new_node = PacketNode(generated_at, data)

        # Case 1: Empty List
        if self.head is None:
            self.head = new_node
            self.tail = new_node
            self.size += 1
            return

        # Case 2: New node is the newest (append to tail) - Most common case!
        if generated_at >= self.tail.generated_at:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
            self.size += 1
            return

        # Case 3: Out-of-order packet (traverse backward from tail)
        current = self.tail
        while current is not None and current.generated_at > generated_at:
            current = current.prev

        if current is None:
            # Reached the very beginning (new head)
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        else:
            # Insert between current and current.next
            new_node.next = current.next
            new_node.prev = current
            if current.next:
                current.next.prev = new_node
            current.next = new_node

        self.size += 1

    def drain_older_than(self, watermark_timestamp: float):
        """
        Pops all nodes from the head that are older than the watermark.
        Returns a list of payload dictionaries in perfect chronological order.
        """
        ready_packets = []
        
        while self.head is not None and self.head.generated_at <= watermark_timestamp:
            ready_packets.append(self.head.data)
            
            # Remove head
            old_head = self.head
            self.head = self.head.next
            if self.head:
                self.head.prev = None
            else:
                self.tail = None # List became empty
                
            old_head.next = None # clear references
            self.size -= 1
            
        return ready_packets

    def trim_older_than(self, expiry_timestamp: float):
        """
        Pops and discards all nodes from the head that are older than the expiry timestamp.
        This prevents memory leaks if the consumer worker crashes.
        Returns the number of packets dropped.
        """
        dropped_count = 0
        while self.head is not None and self.head.generated_at <= expiry_timestamp:
            old_head = self.head
            self.head = self.head.next
            if self.head:
                self.head.prev = None
            else:
                self.tail = None # List became empty
                
            old_head.next = None
            self.size -= 1
            dropped_count += 1
            
        return dropped_count


class ResidentBufferManager:
    """
    Manages a separate Linked List queue for every resident.
    """
    def __init__(self, delay_watermark_seconds=30.0, max_packet_age_seconds=3600.0):
        self.buffers = {}
        self.delay_watermark_seconds = delay_watermark_seconds
        self.max_packet_age_seconds = max_packet_age_seconds

    def add_packet(self, resident_id: int, generated_at_unix: float, packet_data: dict):
        if resident_id not in self.buffers:
            self.buffers[resident_id] = TimeSeriesLinkedList()
            
        self.buffers[resident_id].insert(generated_at_unix, packet_data)
        
        # Every time we insert, we auto-trim extremely old packets to prevent memory leaks
        expiry_cutoff = time.time() - self.max_packet_age_seconds
        self.buffers[resident_id].trim_older_than(expiry_cutoff)

    def process_ready_packets(self, resident_id: int):
        if resident_id not in self.buffers:
            return []
            
        watermark = time.time() - self.delay_watermark_seconds
        return self.buffers[resident_id].drain_older_than(watermark)
        
    def get_queue_size(self, resident_id: int):
        if resident_id in self.buffers:
            return self.buffers[resident_id].size
        return 0
        
    def cleanup_inactive_buffers(self, inactivity_timeout_seconds=86400.0):
        """
        If a resident's queue is completely empty and hasn't been used for 24h,
        delete the queue object entirely to free memory.
        (Implementation simplified: just removes empty queues).
        """
        to_delete = [res_id for res_id, buf in self.buffers.items() if buf.size == 0]
        for res_id in to_delete:
            del self.buffers[res_id]
