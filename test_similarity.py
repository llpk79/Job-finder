import spacy
from sklearn.neighbors import NearestNeighbors
from examples import first, second, third, fourth, fifth, sixth, seventh
nlp = spacy.load('en_core_web_lg')

a = 'Gardening and golf are my favorite things.'
b = 'I like to read and paint for fun.'
c = 'Stocks rose five percent on Saturday.'
d = 'Are you able to perform other duties as assigned?'
e = 'Susan gave the manager a look that could kill.'
f = 'Fly-fishing is the hobby I like best.'

a1 = nlp(first)
b1 = nlp(second)
c1 = nlp(third)
d1 = nlp(fourth)
e5 = nlp(fifth)
f6 = nlp(sixth)
g7 = nlp(seventh)

print(f6.similarity(a1))
print(f6.similarity(b1))
print(f6.similarity(c1))
print(f6.similarity(d1))
print(f6.similarity(e5))
print()
print(g7.similarity(a1))
print(g7.similarity(b1))
print(g7.similarity(c1))
print(g7.similarity(d1))
print(g7.similarity(e5))


nn = NearestNeighbors(5, algorithm='kd_tree')

strings = [first, second, third, fourth, fifth]
X = [nlp(x).vector for x in strings]
nn.fit(X)
print(nn.kneighbors([nlp(sixth).vector]))
print(nn.kneighbors_graph([nlp(sixth).vector]))
print()
print(nn.kneighbors([nlp(seventh).vector]))
print(nn.kneighbors_graph([nlp(seventh).vector]))
