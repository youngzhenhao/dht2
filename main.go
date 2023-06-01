package main

import (
	"bytes"
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"math/rand"
	"time"
)

// 定义地址长度为20字节，每个桶中节点的数量为3个
const (
	AddressLength = 20
	BucketSize    = 3
)

// 定义节点结构，包含ID和地址
type Node struct {
	ID     string  // 使用十六进制字符串表示
	Addr   string  // 使用IP:Port格式表示
	bucket *Bucket // 所属的桶
}

// 定义桶结构，包含前缀、深度和节点列表
type Bucket struct {
	prefix []byte   // 桶的前缀，用于判断节点是否属于该桶
	depth  int      // 桶的深度，用于判断是否需要分裂
	nodes  []*Node  // 桶中的节点列表，按照最近最少使用（LRU）顺序排列
	parent *KBucket // 所属的KBucket
}

// 定义KBucket结构，包含根桶和本地节点ID
type KBucket struct {
	root   *Bucket    // 根桶，覆盖整个ID空间
	local  string     // 本地节点ID，用于计算距离和判断桶是否包含本地节点
	random *rand.Rand // 随机数生成器，用于随机选择节点
}

// 定义Peer结构，包含ID、地址和KBucket
type Peer struct {
	ID     string            // 节点ID
	Addr   string            // 节点地址
	kb     *KBucket          // 节点的KBucket
	values map[string][]byte // 节点存储的键值对
}

// 定义DHT结构，包含所有的Peer和一个通道用于广播新加入的Peer
type DHT struct {
	peers []*Peer    // 所有的Peer列表
	ch    chan *Peer // 通道用于广播新加入的Peer
}

// 计算两个节点ID之间的距离，使用XOR运算并转换为整数值
func distance(id1, id2 string) int {
	b1, _ := hex.DecodeString(id1)
	b2, _ := hex.DecodeString(id2)
	b3 := make([]byte, AddressLength)
	for i := 0; i < AddressLength; i++ {
		b3[i] = b1[i] ^ b2[i]
	}
	d := 0
	for _, b := range b3 {
		d = d*256 + int(b)
	}
	return d
}

// 判断一个节点是否属于一个桶，根据节点ID和桶的前缀是否匹配
func (b *Bucket) contains(node *Node) bool {
	id, _ := hex.DecodeString(node.ID)
	return bytes.HasPrefix(id, b.prefix)
}

// 在一个桶中插入一个节点，如果桶已满则尝试分裂或丢弃，如果桶不满则插入到最前面或更新位置
func (b *Bucket) insert(node *Node) {
	// 遍历桶中的节点列表，查找是否已经存在该节点或需要替换的节点（最后一个）
	var old *Node = nil
	var index int = -1
	for i, n := range b.nodes {
		if n.ID == node.ID {
			index = i // 找到已经存在的节点，记录其位置
			break
		}
		if i == len(b.nodes)-1 {
			old = n // 找到需要替换的节点（最后一个），记录其指针
		}
	}
	if index != -1 { // 如果已经存在该节点，则将其移动到最前面（最近最少使用）
		b.nodes = append(b.nodes[:index], b.nodes[index+1:]...) // 删除该节点
		b.nodes = append([]*Node{node}, b.nodes...)             // 插入到最前面
		node.bucket = b                                         // 更新该节点所属的桶
		return                                                  // 结束插入操作
	}
	if len(b.nodes) < BucketSize { // 如果桶没有满，则直接插入到最前面（最近最少使用）
		b.nodes = append([]*Node{node}, b.nodes...) // 插入到最前面
		node.bucket = b                             // 更新该节点所属的桶
		return                                      // 结束插入操作
	}
	if b.containsLocal() { // 如果桶已满且包含本地节点，则尝试分裂该桶
		b.split()             // 分裂该桶为两个子桶，并重新分配原来的节点到子桶中
		if b.contains(node) { // 如果原来的桶包含待插入的节点，则继续在原来的桶中插入该节点（递归调用）
			b.insert(node)
			return
		} else { // 如果原来的桶不包含待插入的节点，则在新生成的另一个子桶中插入该节点（递归调用）
			b.parent.root.insert(node)
			return
		}
	}
	if old != nil { // 如果桶已满且不包含本地节点，则尝试替换最后一个节点（最近最常使用）
		if old.isAlive() { // 如果最后一个节点仍然存活，则放弃插入操作（丢弃新节点）
			return
		} else { // 如果最后一个节点已经死亡，则用新节点替换它，并移动到最前面（最近最少使用）
			b.nodes[len(b.nodes)-1] = node                                        // 替换最后一个节点为新节点
			b.nodes = append(b.nodes[:len(b.nodes)-1], b.nodes[len(b.nodes):]...) // 删除最后一个节点
			b.nodes = append([]*Node{node}, b.nodes...)                           // 插入到最前面
			node.bucket = b                                                       // 更新新节点所属的桶
			return                                                                // 结束插入操作
		}
	}
}

// 判断一个桶是否包含本地节点，根据本地节点ID和桶的前缀是否匹配
func (b *Bucket) containsLocal() bool {
	local, _ := hex.DecodeString(b.parent.local)
	return bytes.HasPrefix(local, b.prefix)
}

// 分裂一个桶为两个子桶，并重新分配原来的节点到子桶中
func (b *Bucket) split() {
	prefix0 := append(b.prefix, 0) // 生成第一个子桶的前缀，在原来的前缀后加上0
	prefix1 := append(b.prefix, 1) // 生成第二个子桶的前缀，在原来的前缀后加上1
	bucket0 := &Bucket{            // 创建第一个子桶
		prefix: prefix0,
		depth:  b.depth + 1,
		nodes:  make([]*Node, 0, BucketSize),
		parent: b.parent,
	}
	bucket1 := &Bucket{ // 创建第二个子桶
		prefix: prefix1,
		depth:  b.depth + 1,
		nodes:  make([]*Node, 0, BucketSize),
		parent: b.parent,
	}
	for _, node := range b.nodes { // 遍历原来的桶中的所有节点
		if bucket0.contains(node) { // 如果第一个子桶包含该节点，则将其插入到第一个子桶中
			bucket0.insert(node)
		} else { // 如果第二个子桶包含该节点，则将其插入到第二个子桶中
			bucket1.insert(node)
		}
	}
	if len(bucket0.nodes) == 0 { // 如果第一个子桶为空，则将其删除，并将第二个子桶替换为原来的桶
		b.prefix = bucket1.prefix
		b.depth = bucket1.depth
		b.nodes = bucket1.nodes
		for _, node := range b.nodes {
			node.bucket = b
		}
	} else if len(bucket1.nodes) == 0 { // 如果第二个子桶为空，则将其删除，并将第一个子桶替换为原来的桶
		b.prefix = bucket0.prefix
		b.depth = bucket0.depth
		b.nodes = bucket0.nodes
		for _, node := range b.nodes {
			node.bucket = b
		}
	} else { // 如果两个子桶都不为空，则将原来的桶替换为第一个子桶，并将第二个子桶作为新的根桶
		b.parent.root = bucket1
		bucket1.parent.root = bucket0
		b.prefix = bucket0.prefix
		b.depth = bucket0.depth
		b.nodes = bucket0.nodes
		for _, node := range b.nodes {
			node.bucket = b
		}
	}
}

// 在KBucket中插入一个节点，根据节点ID找到对应的桶并调用桶的插入方法
func (kb *KBucket) insertNode(node *Node) {
	kb.root.insert(node) // 从根桶开始插入节点
}

// 打印KBucket中每个桶的内容，遍历所有的桶并打印桶的前缀和节点ID
func (kb *KBucket) printBucketContents() {
	var print func(b *Bucket) // 定义一个递归函数，用于打印一个桶及其子桶的内容
	print = func(b *Bucket) {
		if b == nil || b == b.parent.root { // 如果桶为空，则直接返回
			return
		}
		fmt.Printf("Bucket prefix: %x\n", b.prefix) // 打印桶的前缀
		fmt.Println("Bucket nodes:")                // 打印桶的节点
		for _, node := range b.nodes {
			fmt.Println(node.ID)
		}
		fmt.Println()        // 换行
		print(b.parent.root) // 递归打印另一个子桶的内容
	}
	print(kb.root) // 从根桶开始打印
}

// 判断一个节点是否存活，暂时使用随机数模拟，有一定概率返回false
func (node *Node) isAlive() bool {
	return rand.Intn(10) != 0 // 有十分之一的概率返回false，表示节点已经死亡
}

// 在KBucket中查找一个节点，根据节点ID找到对应的桶并遍历桶中的节点列表，如果找到则返回true，否则返回false
func (kb *KBucket) findNode(nodeID string) bool {
	b := kb.findBucket(nodeID) // 根据节点ID找到对应的桶
	if b == nil {              // 如果没有找到对应的桶，则返回false
		return false
	}
	for _, node := range b.nodes { // 遍历桶中的节点列表
		if node.ID == nodeID { // 如果找到匹配的节点，则返回true
			return true
		}
	}
	return false // 如果没有找到匹配的节点，则返回false
}

// 根据节点ID找到对应的桶，从根桶开始遍历所有的子桶，直到找到包含该节点ID的桶或者没有更多子桶为止
func (kb *KBucket) findBucket(nodeID string) *Bucket {
	b := kb.root                      // 从根桶开始查找
	id, _ := hex.DecodeString(nodeID) // 将节点ID转换为字节切片
	for {                             // 循环查找子桶
		if b.containsLocal() { // 如果当前桶包含本地节点，则说明有两个子桶
			if bytes.HasPrefix(id, append(b.prefix, 0)) { // 如果节点ID匹配第一个子桶的前缀，则继续在第一个子桶中查找
				b = b.parent.root
			} else if bytes.HasPrefix(id, append(b.prefix, 1)) { // 如果节点ID匹配第二个子桶的前缀，则继续在第二个子桶中查找
				b = b.parent.root
			} else { // 如果节点ID不匹配任何子桶的前缀，则说明没有对应的桶，返回nil
				return nil
			}
		} else { // 如果当前桶不包含本地节点，则说明没有更多子桶，返回当前桶
			return b
		}
	}
}

// 创建一个新的Peer，根据地址生成ID，并初始化KBucket和键值对映射表
func newPeer(addr string) *Peer {
	id := sha1.Sum([]byte(addr)) // 使用SHA-1算法生成ID
	p := &Peer{                  // 创建Peer结构体
		ID:   hex.EncodeToString(id[:]), // 将ID转换为十六进制字符串表示
		Addr: addr,                      // 设置地址为给定参数
		kb: &KBucket{ // 初始化KBucket结构体
			root: &Bucket{ // 初始化根桶结构体
				prefix: make([]byte, 0),              // 设置根桶前缀为空切片
				depth:  0,                            // 设置根桶深度为0
				nodes:  make([]*Node, 0, BucketSize), // 设置根桶节点列表为空切片，容量为3
				parent: nil,                          // 设置根桶父指针为nil
			},
			local:  hex.EncodeToString(id[:]),                       // 设置本地节点ID为生成的ID
			random: rand.New(rand.NewSource(time.Now().UnixNano())), // 初始化随机数生成器
		},
		values: make(map[string][]byte), // 初始化键值对映射表为空映射
	}
	p.kb.root.parent = p.kb // 设置根桶父指针为KBucket结构体
	return p                // 返回创建好的Peer结构体
}

// 在DHT中加入一个新的Peer，并广播给其他Peer
func (dht *DHT) join(peer *Peer) {
	dht.peers = append(dht.peers, peer) // 将新Peer加入到DHT中
	dht.ch <- peer                      // 将新Peer发送到通道中
	for _, p := range dht.peers {       // 遍历DHT中所有的Peer
		if p != peer { // 如果不是新加入的Peer
			p.kb.insertNode(&Node{ // 则在其KBucket中插入新Peer作为一个节点
				ID:   peer.ID,
				Addr: peer.Addr,
			})
		}
	}
}

// 在DHT中创建一个新的Peer，并通过其中一个已存在的Peer加入
func (dht *DHT) createAndJoin(addr string) {
	peer := newPeer(addr)   // 创建一个新的Peer
	if len(dht.peers) > 0 { // 如果DHT中已经有其他Peer
		other := dht.peers[rand.Intn(len(dht.peers))] // 则随机选择其中一个
		other.kb.insertNode(&Node{                    // 并在其KBucket中插入新Peer作为一个节点
			ID:   peer.ID,
			Addr: peer.Addr,
		})
	}
	dht.join(peer) // 然后让新Peer加入到DHT中
}

// 在DHT中查找一个节点，如果存在则返回true，否则从对应的桶中随机选择两个节点并发送FindNode请求，并返回false
func (dht *DHT) findNode(peer *Peer, nodeID string) bool {
	peer.kb.insertNode(&Node{
		ID:   nodeID,
		Addr: "",
	})
	if peer.kb.findNode(nodeID) {
		return true
	} else {
		b := peer.kb.findBucket(nodeID)
		if b == nil {
			return false
		}
		n := len(b.nodes)
		if n > 2 {
			n = 2
		}
		for _, i := range rand.Perm(n)[:n] {
			node := b.nodes[i]
			fmt.Printf("Sending FindNode(%s) to %s\n", nodeID, node.Addr)
			dht.ch <- &Peer{
				ID:   node.ID,
				Addr: node.Addr,
			}
		}
		return false
	}
}

// 在DHT中设置一个键值对，如果键是值的哈希则保存并广播给距离最近的两个节点，否则返回false
func (dht *DHT) setValue(peer *Peer, key, value []byte) bool {
	hash := sha1.Sum(value)
	if !bytes.Equal(key, hash[:]) {
		return false
	}
	keyStr := hex.EncodeToString(key)
	if _, ok := peer.values[keyStr]; ok {
		return true
	}
	peer.values[keyStr] = value
	b := peer.kb.findBucket(keyStr)
	if b == nil {
		return true
	}
	n := len(b.nodes)
	if n > 2 {
		n = 2
	}
	for _, i := range rand.Perm(n)[:n] {
		node := b.nodes[i]
		fmt.Printf("Sending SetValue(%x, %s) to %s\n", key, value, node.Addr)
		dht.ch <- &Peer{
			ID:   node.ID,
			Addr: node.Addr,
		}
	}
	return true
}

// 在DHT中获取一个键对应的值，如果本地存在则返回，否则向距离最近的两个节点发送GetValue请求，并等待返回
func (dht *DHT) getValue(peer *Peer, key []byte) []byte {
	keyStr := hex.EncodeToString(key)
	if value, ok := peer.values[keyStr]; ok {
		return value
	}
	b := peer.kb.findBucket(keyStr)
	if b == nil {
		return nil
	}
	n := len(b.nodes)
	if n > 2 {
		n = 2
	}
	for _, i := range rand.Perm(n)[:n] {
		node := b.nodes[i]
		fmt.Printf("Sending GetValue(%x) to %s\n", key, node.Addr)
		dht.ch <- &Peer{
			ID:   node.ID,
			Addr: node.Addr,
		}
	}
	select {
	case p := <-dht.ch:
		if value, ok := p.values[keyStr]; ok {
			return value
		} else {
			return nil
		}
	case <-time.After(5 * time.Second):
		return nil
	}
}

// 创建一个新的DHT，初始化Peer列表和通道
func newDHT() *DHT {
	return &DHT{
		peers: make([]*Peer, 0),
		ch:    make(chan *Peer),
	}
}

// 测试代码
func main() {
	dht := newDHT() // 创建一个新的DHT
	dht.ch = make(chan *Peer, 3000000)
	// 初始化5个Peer
	dht.createAndJoin("127.0.0.1:10001")
	dht.createAndJoin("127.0.0.1:10002")
	dht.createAndJoin("127.0.0.1:10003")
	dht.createAndJoin("127.0.0.1:10004")
	dht.createAndJoin("127.0.0.1:10005")
	// 打印这5个Peer的桶信息
	for _, p := range dht.peers {
		fmt.Printf("Peer %s:\n", p.ID)
		p.kb.printBucketContents()
	}
	// 生成200个新的Peer，并通过之前的5个节点加入到DHT中
	for i := 6; i <= 205; i++ {
		addr := fmt.Sprintf("127.0.0.1:10%03d", i)
		dht.createAndJoin(addr)
	}
	// 打印这205个节点每个节点的桶信息
	for _, p := range dht.peers {
		fmt.Printf("Peer %s:\n", p.ID)
		p.kb.printBucketContents()
	}
	// 随机生成200个字符串，并计算其哈希作为键，然后随机从205个节点中选出一个执行SetValue操作
	keys := make([][]byte, 0, 200)
	for i := 0; i < 200; i++ {
		value := []byte(fmt.Sprintf("value%d", i))
		key := sha1.Sum(value)
		keys = append(keys, key[:])
		p := dht.peers[rand.Intn(len(dht.peers))]
		fmt.Printf("Setting value %s with key %x on peer %s\n", value, key, p.ID)
		dht.setValue(p, key[:], value)
	}
	// 从200个键中随机选择100个，然后每个键再去随机找一个节点执行GetValue操作
	for _, i := range rand.Perm(200)[:100] {
		key := keys[i]
		p := dht.peers[rand.Intn(len(dht.peers))]
		fmt.Printf("Getting value with key %x from peer %s\n", key, p.ID)
		value := dht.getValue(p, key)
		if value != nil {
			fmt.Printf("Got value %s\n", value)
		} else {
			fmt.Println("Got nil")
		}
	}
}
