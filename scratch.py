from infinibatch.iterators import NativeCheckpointableIterator, PrefetchIterator

l = list(range(10))
source = NativeCheckpointableIterator(l)
buffer = PrefetchingIterator(source, buffer_size=4)

for _ in range(6):
    print(next(buffer))

checkpoint = buffer.getstate()
print('chk:', checkpoint)
buffer.setstate(checkpoint)

for _ in range(4):
    print(next(buffer))

checkpoint = buffer.getstate()
print('chk:', checkpoint)
buffer.setstate(checkpoint)
print(next(buffer))