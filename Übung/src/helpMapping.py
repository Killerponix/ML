import numpy as np

def mapYToCluster(Y, cluster):
    classes = np.unique(Y,return_index=False).reshape(-1,) # Alle Klassen in ein Array schreiben [0,1,2]
    
    index = [] # Speicherung aller indizes wo in y classes[0], classes[1], etc. vorkommen.
    
    yClasses = [] # Weise den index[i] zu, welche Klassen denkt das Cluster sind es.
    
    for e in range(len(classes)): # Laufe alle Klassen entlang
        aktIndex = np.where(Y == classes[e])[0] # Schaue, wo überall die Klasse: classes[e] vorkommt.
        index.append(aktIndex) # Speichere in index.
        indexClusterNew, countNew = np.unique(cluster[index[e]], return_counts=True) # Zähle welche und wie oft die Klasse im Indexbereich index[e] vorkommt.
        yClasses.append(indexClusterNew[np.argmax(countNew)]) # Speichert die Häufigste Klasse von cluster[index[e]]
    
    index = np.asarray(index)
    
    if len(yClasses) == len(np.unique(yClasses)): # Wenn keine doppelindizierungen vorkommen.
        
        print("Unique")
        
        for e in range(len(classes)): # Weise Y die Klassen zu, welche das Cluster definiert hat.
            Y[index[e]] = yClasses[e] 
        
        return Y # Gebe Y mit den vom Cluster definierten Klassen zurück
        
    else: # Wenn doppelindizierungen vorkommen:
        
        classes = np.unique(cluster,return_index=False).reshape(-1,)
        
        # Speicher, welche indizes in yClasses vorkommen und wie oft
        unique, counts = np.unique(yClasses, return_counts=True) 
        
        nieZugeordnet = np.delete(classes, unique) # Welche indizes von classes wurden noch nie Zugewiesen?

        #IndexSpeicherung von Klassen, welche doppelt vor kommen. von yClasses
        #z.B. yClasses = [0,1,1] -> zutreffenAufIndex = [1,2] -> Indexe, wo doppelte vorkommen.
        zutreffenAufIndex = np.where(yClasses == unique[np.argmax(counts)])[0]
        
        print(zutreffenAufIndex)
        
        for j in nieZugeordnet: # Laufe alle freien durch von nieZugeordnet
            bestfit = [] # Speichere den bestfit jedes indexBereiches index[e]
            for e in zutreffenAufIndex:#-> Greife auf index[zutreffenAufIndex[i]] zu und schaue
            #in welchem die meisten j vorkommen nieZugeordnet
                #Schaue wie viel % in diesem Indexbereich der Klasse j zugeordnet werden
                bestfit.append(sum(cluster[index[e]] == j)/len(cluster)) 
            # Speichere in dem bestfit die Klasse  nieZugeordnet(j) weil die am besten Zutrifft.
            yClasses[zutreffenAufIndex[np.argmax(bestfit)]] = int(j)
        
        for e in range(len(classes)): # Weise Y die Klassen zu, welche das Cluster definiert hat.
            Y[index[e]] = yClasses[e] 
        
        return Y # Gebe Y mit den vom Cluster definierten Klassen zurück