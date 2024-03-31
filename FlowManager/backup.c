#include <stdint.h>
#include <stdlib.h>
#include <inttypes.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_cycles.h>
#include <rte_lcore.h>
#include <rte_mbuf.h>
#include <rte_mbuf.h>
#include <rte_malloc.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_memzone.h>
#include <rte_mempool.h>
#include <rte_timer.h>
#include <rte_hash.h>
#include <rte_jhash.h>
#include <sys/time.h>
#include <rte_ring.h>

struct rte_ring *ring;
unsigned ring_size = 2048; // 可以根据需要调整大小
const unsigned flags = RING_F_SP_ENQ | RING_F_SC_DEQ; // 单生产者和单消费者标志，如果有多个消费者，需要调整


#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024

#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 32

#define PRINT_IP_ADDR(ip_addr) printf("IP: %d.%d.%d.%d\n",           \
									  (int)((ip_addr) >> 24) & 0xFF, \
									  (int)((ip_addr) >> 16) & 0xFF, \
									  (int)((ip_addr) >> 8) & 0xFF,  \
									  (int)(ip_addr) & 0xFF)

#define PRINT_PORT_MAC(ether_addr)                      \
	printf("MAC: %02" PRIx8 " %02" PRIx8 " %02" PRIx8   \
		   " %02" PRIx8 " %02" PRIx8 " %02" PRIx8 "\n", \
		   RTE_ETHER_ADDR_BYTES(&ether_addr))

#define PRINT_PORT(port) \
	printf("PORT: %" PRIu16 "\n", port);

struct timespec global_time_start;
struct timespec global_time_end;



static uint64_t Interval;
uint64_t total = 0;
uint64_t total_last = 0;

uint64_t Flow_num = 100000;

// 此处指定特征的规格
int packet = 10;
int Len = 1000;
struct rte_mempool *mbuf_pool;
struct rte_mempool *new_mempool;
typedef struct v4_tuple
{
	uint32_t src_addr;
	uint32_t dst_addr;
	union
	{
		struct
		{
			uint16_t sport;
			uint16_t dport;
		};
		uint32_t ports;
	};
} v4_tuple;


//现在用了4-5核心，所以是5
struct rte_hash *flow_tables[8];

typedef struct flow
{
	int limit;
} flow;

flow *buckets[8];




static struct rte_hash *create_hash_table(const char *tablename, uint32_t entry_num, uint32_t init_num)
{
	struct rte_hash *hash_table;
	struct rte_hash_parameters hash_params = {0};

	hash_params.entries = entry_num;
	hash_params.key_len = sizeof(v4_tuple);
	hash_params.hash_func = rte_jhash;
	hash_params.hash_func_init_val = init_num;

	hash_params.name = tablename;
	hash_params.socket_id = rte_socket_id();
	hash_params.extra_flag = RTE_HASH_EXTRA_FLAGS_RW_CONCURRENCY;
	// hash_params.key_mode = RTE_HASH_KEY_MODE_DUP;

	/* Find if the hash table was created before */
	hash_table = rte_hash_find_existing(hash_params.name);
	if (hash_table != NULL)
	{
		printf("exist\n");
		return hash_table;
	}
	else
	{
		hash_table = rte_hash_create(&hash_params);
		if (!hash_table)
		{
			printf("creay failed\n");
			exit(0);
		}
	}

	return hash_table;
}

// 启动端口
static inline int
port_init(uint16_t port, struct rte_mempool *mbuf_pool)
{
//	struct rte_eth_conf port_conf;
	const uint16_t rx_rings = rte_lcore_count()-1, tx_rings = rte_lcore_count()-1;
	uint16_t nb_rxd = RX_RING_SIZE;
	uint16_t nb_txd = TX_RING_SIZE;
	int retval;
	uint16_t q;
	struct rte_eth_dev_info dev_info;
	struct rte_eth_txconf txconf;

	if (!rte_eth_dev_is_valid_port(port))
		return -1;

//	memset(&port_conf, 0, sizeof(struct rte_eth_conf));

	struct rte_eth_conf port_conf = {
		.rxmode = {
			.mq_mode = RTE_ETH_MQ_RX_RSS,
		},
		.rx_adv_conf = {
			.rss_conf = {
				.rss_key = NULL,
				.rss_hf = RTE_ETH_RSS_IP | RTE_ETH_RSS_UDP | RTE_ETH_RSS_TCP,
			},
		},
	};

	retval = rte_eth_dev_info_get(port, &dev_info);
	if (retval != 0)
	{
		printf("Error during getting device (port %u) info: %s\n",
			   port, strerror(-retval));
		return retval;
	}

	if (dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE)
		port_conf.txmode.offloads |=
			RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE;

	/* Configure the Ethernet device. */
	retval = rte_eth_dev_configure(port, rx_rings, tx_rings, &port_conf);
	if (retval != 0)
		return retval;

	retval = rte_eth_dev_adjust_nb_rx_tx_desc(port, &nb_rxd, &nb_txd);
	if (retval != 0)
		return retval;

	/* Allocate and set up 1 RX queue per Ethernet port. */
	for (q = 0; q < rx_rings; q++)
	{
		retval = rte_eth_rx_queue_setup(port, q, nb_rxd,
										rte_eth_dev_socket_id(port), NULL, mbuf_pool);
		if (retval < 0)
			return retval;
	}

	txconf = dev_info.default_txconf;
	txconf.offloads = port_conf.txmode.offloads;
	/* Allocate and set up 1 TX queue per Ethernet port. */
	for (q = 0; q < tx_rings; q++)
	{
		retval = rte_eth_tx_queue_setup(port, q, nb_txd,
										rte_eth_dev_socket_id(port), &txconf);
		if (retval < 0)
			return retval;
	}

	/* Starting Ethernet port. 8< */
	retval = rte_eth_dev_start(port);
	/* >8 End of starting of ethernet port. */
	if (retval < 0)
		return retval;

	/* Display the port MAC address. */
	struct rte_ether_addr addr;
	retval = rte_eth_macaddr_get(port, &addr);
	if (retval != 0)
		return retval;

	PRINT_PORT_MAC(addr);

	/* Enable RX in promiscuous mode for the Ethernet device. */
	retval = rte_eth_promiscuous_enable(port);
	/* End of setting RX port in promiscuous mode. */
	if (retval != 0)
		return retval;

	return 0;
}

uint64_t cycle_diff[10];
float totalMilliseconds[10] = {0};

static int
lcore_main(__rte_unused void *arg)
{
	uint16_t port;
	struct rte_ipv4_hdr *ipv4;
	struct rte_tcp_hdr *tcp;
	struct rte_udp_hdr *udp;
	v4_tuple ipv4_tuple;
	struct rte_ether_hdr *eth_hdr;

	static int is_time_initialized = 0;

	RTE_ETH_FOREACH_DEV(port)
	if (rte_eth_dev_socket_id(port) >= 0 &&
		rte_eth_dev_socket_id(port) !=
			(int)rte_socket_id())
		printf("WARNING, port %u is on remote NUMA node to "
			   "polling thread.\n\tPerformance will "
			   "not be optimal.\n",
			   port);

	printf("\nCore %u forwarding packets. [Ctrl+C to quit]\n", rte_lcore_id());
	int index = rte_lcore_id() - 1;
	/* Main work of application loop. 8< */
	while (1)
	{
		for(port = 0; port < 2; port++)
		{
			/* Get burst of RX packets, from first port of pair. */
			struct rte_mbuf *bufs[BURST_SIZE];
			const uint16_t nb_rx = rte_eth_rx_burst(port, index,
													bufs, BURST_SIZE);

			if (unlikely(nb_rx == 0))
				continue;

			/* Send burst of TX packets, to second port of pair. */
			// const uint16_t nb_tx = rte_eth_tx_burst(port, 0,bufs, nb_rx);
			if (nb_rx > 0)
			{

				total += nb_rx;
					//printf("Total %ld\n",total);
			}

			/*这一块用于释放收包*/
			uint16_t buf;
			for (buf = 0; buf < nb_rx; buf++)
			{

				ipv4 = (struct rte_ipv4_hdr *)(rte_pktmbuf_mtod(bufs[buf], struct rte_ether_hdr *) + 1);
				ipv4_tuple.src_addr = rte_be_to_cpu_32(ipv4->src_addr);
				ipv4_tuple.dst_addr = rte_be_to_cpu_32(ipv4->dst_addr);
				eth_hdr = rte_pktmbuf_mtod(bufs[buf], struct rte_ether_hdr *);
				

				/* only process TCP flows */
				if (ipv4->next_proto_id == 6)
				{
					//printf("hash %d\n",rte_lcore_id());
					uint64_t start_time = rte_get_timer_cycles();
					tcp = (struct rte_tcp_hdr *)((unsigned char *)ipv4 + sizeof(struct rte_ipv4_hdr));
					ipv4_tuple.sport = rte_be_to_cpu_16(tcp->src_port);
					ipv4_tuple.dport = rte_be_to_cpu_16(tcp->dst_port);

					int ret = rte_hash_lookup(flow_tables[index], (void const *)&ipv4_tuple);

					if (ret == -ENOENT)
					{
						/* new flow */
						ret = rte_hash_add_key(flow_tables[index], (void const *)&ipv4_tuple);
						buckets[index][ret].limit++;
						
//						printf("queue%d flowLimit%d firstPacket%d\n",index,packet,buckets[index][ret].limit);
					}
					else if (ret >= 0)
					{
						buckets[index][ret].limit++;
					}

						
					//这里根据提取特征的规格可以修改
					if(buckets[index][ret].limit <= packet){
						struct rte_mbuf *cloned_mbuf = rte_pktmbuf_clone(bufs[buf], new_mempool);
						if (cloned_mbuf != NULL) {
							int ret = rte_ring_enqueue(ring, (void *)cloned_mbuf);
							if (ret < 0) {
								// 入队失败，处理错误
								rte_pktmbuf_free(cloned_mbuf); // 释放 mbuf 以避免内存泄漏
								printf("Ring is full, mbuf dropped\n");
							}
						}					
					}
										
					//unsigned char *payload = rte_pktmbuf_mtod_offset(bufs[buf], unsigned char *, 54);
				}
				const uint16_t nb_tx = rte_eth_tx_burst(port^1, index,
				&bufs[buf], 1);
			}
		}
	}
	/* >8 End of loop. */
	return 0;
}

static void stats_display()
{
	const char clr[] = {27, '[', '2', 'J', '\0'};
	const char top_left[] = {27, '[', '1', ';', '1', 'H', '\0'};
	int i, portid;

	/* Clear screen and move to top left */
	printf("%s%s", clr, top_left);
	printf("Total %ld\n", total - total_last);
	total_last = total;
}



int main(int argc, char *argv[])
{

	unsigned nb_ports;
	uint16_t portid;

	/* Initializion the Environment Abstraction Layer (EAL). 8< */
	int ret = rte_eal_init(argc, argv);
	if (ret < 0)
		rte_exit(EXIT_FAILURE, "Error with EAL initialization\n");
	/* >8 End of initialization the Environment Abstraction Layer (EAL). */

	argc -= ret;
	argv += ret;
	if (argc > 1) {
        // 假设第一个非EAL参数是您的整数
        packet = atoi(argv[1]);
    } 


	/* Check that there is an even number of ports to send/receive on. */
	nb_ports = rte_eth_dev_count_avail();
	mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS * 10,
										MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
	/* >8 End of allocating mempool to hold mbuf. */
	ring = rte_ring_create("mbuf_ring", ring_size, rte_socket_id(), flags);
	if (ring == NULL) {
		rte_exit(EXIT_FAILURE, "Cannot create ring\n");
	}

	if (mbuf_pool == NULL)
		rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");

	unsigned num_mbufs = 1024; // 内存池中mbuf的数量，根据需要调整
	new_mempool = rte_pktmbuf_pool_create("new_mempool", num_mbufs,
										250, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());

	if (new_mempool == NULL) {
		rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");
	}


	/* Initializing all ports. 8< */
	RTE_ETH_FOREACH_DEV(portid)
	if (port_init(portid, mbuf_pool) != 0)
		rte_exit(EXIT_FAILURE, "Cannot init port %" PRIu16 "\n",
				 portid);
	/* >8 End of initializing all ports. */

	buckets[0] = (flow *)rte_malloc("bucket", Flow_num * sizeof(flow), RTE_CACHE_LINE_SIZE);
	buckets[1] = (flow *)rte_malloc("bucket", Flow_num * sizeof(flow), RTE_CACHE_LINE_SIZE);	
	buckets[2] = (flow *)rte_malloc("bucket", Flow_num * sizeof(flow), RTE_CACHE_LINE_SIZE);
	buckets[3] = (flow *)rte_malloc("bucket", Flow_num * sizeof(flow), RTE_CACHE_LINE_SIZE);
	buckets[4] = (flow *)rte_malloc("bucket", Flow_num * sizeof(flow), RTE_CACHE_LINE_SIZE);
	buckets[5] = (flow *)rte_malloc("bucket", Flow_num * sizeof(flow), RTE_CACHE_LINE_SIZE);
	buckets[6] = (flow *)rte_malloc("bucket", Flow_num * sizeof(flow), RTE_CACHE_LINE_SIZE);
	buckets[7] = (flow *)rte_malloc("bucket", Flow_num * sizeof(flow), RTE_CACHE_LINE_SIZE);


	const int TABLE_COUNT = 8; // 假设您有8个表
	for (int i = 0; i < TABLE_COUNT; i++) {
		char name[20]; // 分配足够的空间以容纳表名和数字
		snprintf(name, sizeof(name), "my_table%d", i); // 创建表名，如 my_table0, my_table1, 等。
		flow_tables[i] = create_hash_table(name, Flow_num, 1);
	}

	printf("\n%d lcores used.\n", rte_lcore_count());

	// 初始化定时器子系统
	rte_timer_subsystem_init();

	// 创建定时器
	struct rte_timer timer;
	rte_timer_init(&timer);

	// 设置定时器每秒触发一次
	uint64_t hz = rte_get_timer_hz();
	Interval = hz;


	for(int z = 1;z < rte_lcore_count(); z++)
	{
		rte_eal_remote_launch(lcore_main, NULL, z);
	}

	struct rte_mbuf *mbufs_to_send[32]; // 用于存储指向mbuf的指针

	while (1) {
		// 从环形缓冲区中批量取出mbuf
		unsigned int num_dequeued = rte_ring_dequeue_burst(ring, (void **)mbufs_to_send, 32, NULL);

		if (num_dequeued > 0) {
			// 成功从环形缓冲区取出mbuf，现在批量发送
			const uint16_t nb_tx = rte_eth_tx_burst(3, 0, mbufs_to_send, num_dequeued);
			if (nb_tx < num_dequeued) {
				// 如果有mbuf没有被发送，需要处理它们（例如释放或重试）
				for (uint16_t i = nb_tx; i < num_dequeued; i++) {
					rte_pktmbuf_free(mbufs_to_send[i]);
				}
			}
		}
		// 可以添加适当的延迟或条件以避免忙等
	}

	rte_eal_mp_wait_lcore();
	rte_eal_cleanup();
	return 0;
}