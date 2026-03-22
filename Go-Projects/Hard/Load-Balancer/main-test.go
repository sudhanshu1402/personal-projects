package main

import (
	"net/url"
	"testing"
)

func TestServerPool_AddBackend(t *testing.T) {
	pool := &ServerPool{backends: []*ServerNode{}}
	u, _ := url.Parse("http://localhost:8080")
	pool.AddBackend(&ServerNode{URL: u, Alive: true})

	if len(pool.backends) != 1 {
		t.Errorf("Expected 1 backend, got %d", len(pool.backends))
	}
}

func TestServerPool_NextIndex(t *testing.T) {
	pool := &ServerPool{backends: []*ServerNode{}}
	u1, _ := url.Parse("http://localhost:8080")
	u2, _ := url.Parse("http://localhost:8081")
	pool.AddBackend(&ServerNode{URL: u1, Alive: true})
	pool.AddBackend(&ServerNode{URL: u2, Alive: true})

	idx1 := pool.NextIndex()
	idx2 := pool.NextIndex()

	if idx1 == idx2 {
		t.Errorf("Round robin failed, got same index %d", idx1)
	}
}

func TestServerNode_IsAlive(t *testing.T) {
    u, _ := url.Parse("http://localhost:8080")
    node := &ServerNode{URL: u, Alive: true}
    if !node.IsAlive() {
        t.Error("Node should be alive")
    }
    node.SetAlive(false)
    if node.IsAlive() {
        t.Error("Node should be dead")
    }
}
