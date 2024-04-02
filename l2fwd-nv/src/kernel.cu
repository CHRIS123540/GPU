/* 
 * Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved. 
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "common.h"
#include "cuda_related.h"
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_udp.h>
#include <rte_tcp.h>
#include <rte_gpudev.h>
#include <iostream>
// #define DEBUG_PRINT 1

__device__ __forceinline__ unsigned long long __globaltimer()
{
    unsigned long long globaltimer;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer));
    return globaltimer;
}

/////////////////////////////////////////////////////////////////////////////////////////
//// Regular CUDA kernel -w 2
/////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_mac_update(DataPacket *d_a,struct rte_gpu_comm_list *comm_list, uint64_t wtime_n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//	uint16_t temp;
	unsigned long long pkt_start;
	uint16_t hash_idx[1024];
	struct rte_tcp_hdr *tcp;

	int pktSize = 0;
	if (idx < comm_list->num_pkts && comm_list->pkt_list[idx].addr != 0) {


		struct rte_ether_hdr *eth = (struct rte_ether_hdr *)(((uint8_t *) (comm_list->pkt_list[idx].addr)));

		struct rte_ipv4_hdr *ipv4 = (struct rte_ipv4_hdr *)(eth + 1);
		uint8_t *dst_addr_bytes = (uint8_t *)(&ipv4->dst_addr);
		uint8_t *src_addr_bytes = (uint8_t *)(&ipv4->src_addr);		
		

		if (ipv4->next_proto_id == 6)
		{
			tcp = (struct rte_tcp_hdr *)(ipv4 + 1);
			uint16_t sport = (tcp->src_port);
			uint16_t dport = (tcp->dst_port);
			uint16_t sport_host = ((sport >> 8) | (sport << 8)) & 0xFFFF;
			uint16_t dport_host = ((dport >> 8) | (dport << 8)) & 0xFFFF;	
			//printf("Source port: %u\n", sport_host);
			//	printf("Num%d %d\n",idx,d_a[idx].id);
			hash_idx[idx] = sport_host;


		}
		

		/*
		printf("Destination IPv4 Address: %d.%d.%d.%d\n", 
		dst_addr_bytes[0], dst_addr_bytes[1], dst_addr_bytes[2], dst_addr_bytes[3]);
		*/
		pktSize = comm_list->pkt_list[idx].size;


	//d_a[idx].id = d_a[idx].id + 100;
	//printf("num %d len%d\n",comm_list->num_pkts,d_a[idx].id);
	//printf("GPU len%d\n",comm_list->pkt_list[idx].size);
	}
	if (comm_list->pkt_list[idx].addr == 0)
	{
		//不一定一定会达到1024个数据包
		hash_idx[idx] = 0;
	}
	
	__shared__ unsigned int a_share[1024];
	__shared__ unsigned int hash_share[1024];
	__shared__ unsigned int seq_share[1024];
	__shared__ unsigned int flow_share[1024];
	__shared__ unsigned int temp[2048];
	__shared__ unsigned int flag_share[1024];
	__shared__ unsigned int feature_share[1024];
	int n = 1024;
	int tid = idx;
	int t = 0,flag1 = 0, flag2 = 0;
	
	//需要知道是哪条流，同时区分数据包
	a_share[tid] = hash_idx[tid]*1024+tid;
	//a_share[tid] = a[tid];
	hash_share[tid] = hash_idx[tid];
	seq_share[tid] = tid + 1;
	flow_share[tid] = 1;//flow列
	//temp不需要初始化，只是一个双指针算法
	flag_share[tid] = 0;
	feature_share[tid] = tid;
	__syncthreads();
    for(int i = 1;i < n; i = i << 1)//从序列长度 1 到 序列长度 n/2，我们的目的是对 一个双调函数排序，所以不用对n
    {
        //判断升序还是降序 0为升 1为降 我们的目的是获得 序列长度为2，所以要除2*i
    	//bool order = (tid / (2*i))%2;
        flag1++;
        bool order = (tid >> flag1)%2;
        //printf("%d %d \n",tid, order);
        flag2 = flag1 - 1;       
        for(int j = i; j >= 1; j = j >> 1)
        {
            if(((tid >> flag2)%2) == 0) //除跳跃的步长 再取模 这是对自身的序列做排序，所以不用*2
            {            
                // 升序 & 出现 前 > 后   || 降序 & 前 < 后 并且在最后一次的时候，没用降序，所以一定要判断边界
                if ((tid + j < n) && (   ((!order) == (a_share[tid] > a_share[tid + j]))    ||   (order == (a_share[tid] < a_share[tid + j]))    ))
                {
                    t = a_share[tid];
                    a_share[tid] = a_share[tid + j];
                    a_share[tid + j] = t;
                
                    t = seq_share[tid];
                    seq_share[tid] = seq_share[tid + j];
                    seq_share[tid + j] = t;

                    t = hash_share[tid];
                    hash_share[tid] = hash_share[tid + j];
                    hash_share[tid + j] = t;
                
                }               
            }
            flag2--;
            __syncthreads();
        }
    }
	    //这里做flow列 邻居节点的处理
		if(tid > 0)
		{
			flow_share[tid] = (hash_share[tid] != hash_share[tid-1]);
		}
		temp[tid] = flow_share[tid];
		__syncthreads();
			
		int in = 1;
		int out = 0;
		//前缀和计算有问题
		if(tid < n)
		{
			for(int i = 1;i < n;i = i<<1)
				{
					in = 1 - in;
					out = 1 - out; 
					int index = i;
					if((tid - index) >= 0)
					{                
						temp[tid + n * out] = temp[tid + n * in] + temp[tid - index + n * in];                    
		//                    temp[tid + out] = temp[tid ] + temp[tid - index ];                
					}
					else
					{
						temp[tid + n * out] = temp[tid + n * in];
					}
					__syncthreads();
				}        
		}
		flow_share[tid] = temp[tid + n * out];
		__syncthreads();
			
	
		int j;
		//计算不同流有多少个数据包
		if(tid > 0)
		{
			if(flow_share[tid] != flow_share[tid - 1])
			{
				j = flow_share[tid];
				flag_share[j] = tid;
	
			}
		}
		__syncthreads();
	   
	   j = flow_share[tid];
	   feature_share[tid] = feature_share[tid] - flag_share[j];
		__syncthreads();

	//	if(flow_share[tid] < 100)
	//	printf("%d %d\n",hash_idx[tid],feature_share[tid]);
	// printf("num %d len%d\n",idx,hash_idx[idx]);
	__threadfence();
	__syncthreads();

	uint8_t *payload = (uint8_t *)(tcp+1);
	int ELength = L < pktSize ? L : pktSize;
	for(int z = 0;z < ELength;z++)
	{
		if(feature_share[tid] < H)
		{
			//某个类别的 第几个数据包 的前L个字节
				d_a[hash_idx[idx]].n[feature_share[tid]][z] = payload[z];		
		}
	}	
	
	if (idx == 0) {
		RTE_GPU_VOLATILE(*(comm_list->status_d)) = RTE_GPU_COMM_LIST_DONE;
		__threadfence_system();


	}
	__syncthreads();
}

void workload_launch_gpu_processing(DataPacket *d_a,struct rte_gpu_comm_list * comm_list, uint64_t wtime_n,
							int cuda_blocks, int cuda_threads, cudaStream_t stream)
{
	assert(cuda_blocks == 1);
	assert(cuda_threads > 0);

	if (comm_list == NULL)
		return;

	CUDA_CHECK(cudaGetLastError());
    cudaEvent_t start, stop;
    float time;

    // 创建事件
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始时间
    cudaEventRecord(start, stream);

	kernel_mac_update <<< cuda_blocks, cuda_threads, 0, stream >>> (d_a,comm_list, 0);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, stream);

    // 等待事件完成
    cudaEventSynchronize(stop);

    // 计算运行时间
    cudaEventElapsedTime(&time, start, stop);

    // 输出结果
    std::cout << "Time for the CUDA function: " << time << " ms" << std::endl;

			
	CUDA_CHECK(cudaGetLastError());
}

/////////////////////////////////////////////////////////////////////////////////////////
//// Persistent CUDA kernel -w 3
/////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_persistent_mac_update(struct rte_gpu_comm_list * comm_list, uint64_t wtime_n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int item_index = 0;
	unsigned long long pkt_start;
	struct rte_ether_hdr *eth;
	uint16_t *src_addr, *dst_addr, temp;
	uint32_t wait_status;
	__shared__ uint32_t wait_status_shared[1];
		
	__syncthreads();

	while (1) {
		if (idx == 0)
		{
			while (1)
			{
				wait_status = RTE_GPU_VOLATILE(comm_list[item_index].status_d[0]);
				if(wait_status != RTE_GPU_COMM_LIST_FREE)
				{
					wait_status_shared[0] = wait_status;
					__threadfence_block();
					break;
				}
			}
		}

		__syncthreads();

		if (wait_status_shared[0] != RTE_GPU_COMM_LIST_READY)
			break;

		if (idx < comm_list[item_index].num_pkts && comm_list->pkt_list[idx].addr != 0) {
			if(wtime_n)
				pkt_start = __globaltimer();

			eth = (struct rte_ether_hdr *)(((uint8_t *) (comm_list[item_index].pkt_list[idx].addr)));
			src_addr = (uint16_t *) (&eth->src_addr);
			dst_addr = (uint16_t *) (&eth->dst_addr);

#ifdef DEBUG_PRINT
			/* Code to verify source and dest of ethernet addresses */
			uint8_t *src = (uint8_t *) (&eth->src_addr);
			uint8_t *dst = (uint8_t *) (&eth->dst_addr);
			printf("Before Swap, Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x:%02x:%02x\n",
				src[0], src[1], src[2], src[3], src[4], src[5],
				dst[0], dst[1], dst[2], dst[3], dst[4], dst[5]);
#endif
			temp = dst_addr[0];
			dst_addr[0] = src_addr[0];
			src_addr[0] = temp;
			temp = dst_addr[1];
			dst_addr[1] = src_addr[1];
			src_addr[1] = temp;
			temp = dst_addr[2];
			dst_addr[2] = src_addr[2];
			src_addr[2] = temp;

#ifdef DEBUG_PRINT
		/* Code to verify source and dest of ethernet addresses */
		printf("After Swap, Source: %x:%x:%x:%x:%x:%x Dest: %x:%x:%x:%x:%x:%x\n",
		       ((uint8_t *) (src_addr))[0], ((uint8_t *) (src_addr))[1],
		       ((uint8_t *) (src_addr))[2], ((uint8_t *) (src_addr))[3],
		       ((uint8_t *) (src_addr))[4], ((uint8_t *) (src_addr))[5],
		       ((uint8_t *) (dst_addr))[0], ((uint8_t *) (dst_addr))[1],
		       ((uint8_t *) (dst_addr))[2], ((uint8_t *) (dst_addr))[3],
		       ((uint8_t *) (dst_addr))[4], ((uint8_t *) (dst_addr))[5]);
#endif
			if(wtime_n)
				while((__globaltimer() - pkt_start) < wtime_n);
		}

		__threadfence();
		__syncthreads();
		
		if (idx == 0) {
			RTE_GPU_VOLATILE(comm_list[item_index].status_d[0]) = RTE_GPU_COMM_LIST_DONE;
			__threadfence_system();
		}

		item_index = (item_index + 1) % MAX_BURSTS_X_QUEUE;
	}
}

void workload_launch_persistent_gpu_processing(struct rte_gpu_comm_list * comm_list,
						uint64_t wtime_n,
						int cuda_blocks, int cuda_threads,
						cudaStream_t stream)
{
	assert(cuda_blocks == 1);
	assert(cuda_threads > 0);
	if (comm_list == NULL)
		return;

	CUDA_CHECK(cudaGetLastError());
	kernel_persistent_mac_update <<< cuda_blocks, cuda_threads, 0, stream >>> (comm_list, wtime_n);
	CUDA_CHECK(cudaGetLastError());
}

/////////////////////////////////////////////////////////////////////////////////////////
//// CUDA GRAPHS kernel -w 4
/////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_graphs_mac_update(struct rte_gpu_comm_list * comm_item_list, uint64_t wtime_n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint16_t temp;
	unsigned long long pkt_start;
	uint32_t wait_status;
	__shared__ uint32_t wait_status_shared[1];

	if (idx == 0)
	{
		while (1)
		{
			wait_status = RTE_GPU_VOLATILE(comm_item_list->status_d[0]);
			if(wait_status != RTE_GPU_COMM_LIST_FREE)
			{
				wait_status_shared[0] = wait_status;
				__threadfence_block();
				break;
			}
		}
	}

	__syncthreads();

	if (wait_status_shared[0] != RTE_GPU_COMM_LIST_READY)
		return;

	if (idx < comm_item_list->num_pkts && comm_item_list->pkt_list[idx].addr != 0) {
		if(wtime_n)
			pkt_start = __globaltimer();

		struct rte_ether_hdr *eth = (struct rte_ether_hdr *)(((uint8_t *) (comm_item_list->pkt_list[idx].addr)));
		uint16_t *src_addr = (uint16_t *) (&eth->src_addr);
		uint16_t *dst_addr = (uint16_t *) (&eth->dst_addr);

#ifdef DEBUG_PRINT
		/* Code to verify source and dest of ethernet addresses */
		uint8_t *src = (uint8_t *) (&eth->src_addr);
		uint8_t *dst = (uint8_t *) (&eth->dst_addr);
		printf
		    ("GRAPHS before Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x:%02x:%02x\n",
		     src[0], src[1], src[2], src[3], src[4], src[5], dst[0],
		     dst[1], dst[2], dst[3], dst[4], dst[5]);
#endif

		temp = dst_addr[0];
		dst_addr[0] = src_addr[0];
		src_addr[0] = temp;
		temp = dst_addr[1];
		dst_addr[1] = src_addr[1];
		src_addr[1] = temp;
		temp = dst_addr[2];
		dst_addr[2] = src_addr[2];
		src_addr[2] = temp;

		if(wtime_n)
			while((__globaltimer() - pkt_start) < wtime_n);

#ifdef DEBUG_PRINT
		/* Code to verify source and dest of ethernet addresses */
		printf("GRAPHS after Source: %x:%x:%x:%x:%x:%x Dest: %x:%x:%x:%x:%x:%x\n",
		       ((uint8_t *) (src_addr))[0], ((uint8_t *) (src_addr))[1],
		       ((uint8_t *) (src_addr))[2], ((uint8_t *) (src_addr))[3],
		       ((uint8_t *) (src_addr))[4], ((uint8_t *) (src_addr))[5],
		       ((uint8_t *) (dst_addr))[0], ((uint8_t *) (dst_addr))[1],
		       ((uint8_t *) (dst_addr))[2], ((uint8_t *) (dst_addr))[3],
		       ((uint8_t *) (dst_addr))[4], ((uint8_t *) (dst_addr))[5]);
#endif
	}

	__threadfence();
	__syncthreads();

	if (idx == 0) {
		RTE_GPU_VOLATILE(*(comm_item_list->status_d)) = RTE_GPU_COMM_LIST_DONE;
		__threadfence_system();
	}
	__syncthreads();
}

void workload_launch_gpu_graph_processing(struct rte_gpu_comm_list * bitem, uint64_t wtime_n,
										int cuda_blocks, int cuda_threads, cudaStream_t stream)
{
	assert(cuda_blocks == 1);
	assert(cuda_threads > 0);

	CUDA_CHECK(cudaGetLastError());
	kernel_graphs_mac_update <<< cuda_blocks, cuda_threads, 0, stream >>> (bitem, wtime_n);
	CUDA_CHECK(cudaGetLastError());
}