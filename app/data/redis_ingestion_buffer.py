import json
import time
import logging

try:
    import redis
except ImportError:
    redis = None
    logging.warning("Redis library not installed. Run: pip install redis")


class RedisBLEBuffer:
    """
    A time-series buffer for BLE packets using Redis Sorted Sets (ZSET).
    This handles out-of-order packets, deduplication, and late arrivals.
    """
    
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        if redis is None:
            raise ImportError("Redis library is required. Run: pip install redis")
            
        self.r = redis.Redis(host=host, port=port, db=db, password=password, decode_responses=True)
        self.prefix = "caremp:ble_buffer:res_"
        
        # How long to keep a resident's entire queue alive in Redis if no new data arrives
        self.QUEUE_TTL_SECONDS = 86400  # 24 hours
        
        # Max age of packets to keep in the queue (in case the background worker crashes)
        self.MAX_PACKET_AGE_SECONDS = 3600  # 1 hour

    def _get_key(self, resident_id):
        return f"{self.prefix}{resident_id}"

    def add_packet(self, resident_id: int, generated_at_unix: float, packet_data: dict):
        """
        Pushes a BLE packet into the resident's sorted queue.
        """
        key = self._get_key(resident_id)
        
        # Convert packet dictionary to a string for Redis storage
        # Sorting the dictionary keys ensures exact duplicates produce the same string
        packet_str = json.dumps(packet_data, sort_keys=True)
        
        pipeline = self.r.pipeline()
        
        # 1. Add to the Sorted Set (ZADD)
        # Score = timestamp. Redis will automatically sort it perfectly chronologically.
        pipeline.zadd(key, {packet_str: generated_at_unix})
        
        # 2. Safety Expiry (TTL) on the entire key
        # If the resident leaves the facility and stops transmitting, clean up their queue after 24 hours.
        pipeline.expire(key, self.QUEUE_TTL_SECONDS)
        
        # 3. Trim the Queue (Maintains the 24-hour sliding window perfectly)
        # We delete packets older than 24 hours (86400 seconds)
        # Note: If running historical replays, set this to 10 years (86400 * 365 * 10)
        cutoff_time = time.time() - 86400 
        pipeline.zremrangebyscore(key, min="-inf", max=cutoff_time)
        
        pipeline.execute()
        
        return True

    def drain_ready_packets(self, resident_id: int, delay_watermark_seconds: float = 30.0):
        """
        Pulls packets from the queue that are older than the watermark delay.
        This provides a grace period for out-of-order packets to arrive and be sorted.
        
        Returns a list of dictionaries (the perfectly ordered packets).
        """
        key = self._get_key(resident_id)
        
        # We only want packets older than (current_time - delay)
        process_up_to_time = time.time() - delay_watermark_seconds
        
        pipeline = self.r.pipeline()
        
        # 1. Fetch all ready packets in chronological order
        pipeline.zrangebyscore(key, min="-inf", max=process_up_to_time)
        
        # 2. Remove those exact packets from the queue so they aren't processed again
        pipeline.zremrangebyscore(key, min="-inf", max=process_up_to_time)
        
        results = pipeline.execute()
        
        raw_packets = results[0]  # The result of zrangebyscore
        
        if not raw_packets:
            return []
            
        # Parse back into Python dictionaries
        chronological_packets = [json.loads(p) for p in raw_packets]
        return chronological_packets

    def get_queue_size(self, resident_id: int) -> int:
        """Returns the total number of packets currently buffered for the resident."""
        key = self._get_key(resident_id)
        return self.r.zcard(key)


# ==========================================
# Example Usage / Test
# ==========================================
if __name__ == "__main__":
    import random
    
    print("Initializing Redis BLE Buffer...")
    try:
        buffer = RedisBLEBuffer()
        # Test connection
        buffer.r.ping()
    except Exception as e:
        print(f"Could not connect to Redis (is it running?): {e}")
        exit(1)
        
    res_id = 999
    now = time.time()
    
    print("\n[1] Simulating out-of-order BLE packet arrival...")
    
    # Packet generated 40 seconds ago (ARRIVES FIRST)
    buffer.add_packet(res_id, now - 40, {"hr": 70, "msg": "Older packet"})
    print("Added packet T-40s")
    
    # Packet generated 45 seconds ago (ARRIVES LATE / OUT OF ORDER)
    buffer.add_packet(res_id, now - 45, {"hr": 75, "msg": "Oldest packet (Late arrival)"})
    print("Added packet T-45s (Late Arrival)")
    
    # Packet generated 10 seconds ago (ARRIVES EARLY, should not be processed yet)
    buffer.add_packet(res_id, now - 10, {"hr": 80, "msg": "Recent packet"})
    print("Added packet T-10s")
    
    print(f"Current Queue Size: {buffer.get_queue_size(res_id)}")
    
    print("\n[2] Draining packets older than 30 seconds...")
    # Using a 30-second watermark delay
    ready_packets = buffer.drain_ready_packets(res_id, delay_watermark_seconds=30.0)
    
    print(f"Retrieved {len(ready_packets)} ready packets:")
    for p in ready_packets:
        print(f"  -> {p['msg']}")
        
    print(f"Remaining Queue Size (Waiting for watermark): {buffer.get_queue_size(res_id)}")
