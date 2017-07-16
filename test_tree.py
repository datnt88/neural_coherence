
from zss import simple_distance, Node

A = (
    Node("f").addkid(Node("a").addkid(Node("h"))
            .addkid(Node("c")
            .addkid(Node("l"))))
        .addkid(Node("e"))
    )
B = (
    Node("b")
        .addkid(Node("a")
        .addkid(Node("d"))
        .addkid(Node("e")
        .addkid(Node("f"))))
        .addkid(Node("c"))
    )

C = (
    Node("1").addkid(Node("2").addkid(Node("3")))
    )




tree1_nodes = {}
tree2_nodes = {}

for i in '1234':
    tree1_nodes[i] = Node(i)
for i in '1234':
    tree2_nodes[i] = Node(i)



#print tree1_nodes['3']

B = tree1_nodes['1'].addkid(tree1_nodes['2'].addkid(tree1_nodes['3']))
A = tree2_nodes['1'].addkid(tree2_nodes['2']).addkid(tree2_nodes['3'])

print A
print "-------------------"
print B
print "-------------------"

print simple_distance(A,B)









#print simple_distance(A, B) 
