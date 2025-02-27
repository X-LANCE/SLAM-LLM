# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import gc
import psutil
import threading

import torch

def byte2gb(x):
    return int(x / 2**30)
# This context manager is used to track the peak memory usage of the process
class MemoryTrace:
    def __enter__(self):
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = byte2gb(torch.npu.memory_allocated())
        self.process = psutil.Process()
        self.cpu_begin = byte2gb(self.cpu_mem_used())
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.npu.empty_cache()
        self.end = byte2gb(torch.npu.memory_allocated())
        self.peak = byte2gb(torch.npu.max_memory_allocated())
        npu_info = torch.npu.memory_stats()
        self.peak_active_gb = byte2gb(npu_info["active_bytes.all.peak"])
        self.npu_malloc_retires = npu_info.get("num_alloc_retries", 0)
        self.peak_active_gb = byte2gb(npu_info["active_bytes.all.peak"])
        self.m_npu_ooms = npu_info.get("num_ooms", 0)
        self.used = byte2gb(self.end - self.begin)
        self.peaked = byte2gb(self.peak - self.begin)
        self.max_reserved = byte2gb(torch.npu.max_memory_reserved())

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = byte2gb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = byte2gb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")