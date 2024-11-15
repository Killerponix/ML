foo = 21
if foo == 42:
    print("This is the answer")
elif foo == 21:
    print('At least ...')
    print('it is half of the truth')
else:
    print("I'm sorry, Dave, I'm afraid I can't do that.")
    
wurzel = foo
while abs(wurzel**2 - foo) > 10**-7 :
    wurzel = 0.5 * (wurzel + foo/wurzel)
    
print("Die Wurzel von %e ist %e" % (foo,wurzel) )
 
autoren = ["Douglas Adams","Isaac Asimov", "Terry Pratchett", "Iain Banks"]
for name in autoren:
    print(name)
    
for i in range(1,11,2):
    print(i)

for i, autor in enumerate(autoren):
    print(i, autor)
