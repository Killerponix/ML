def mystemp():
    """ Gibt den aktuellen Zeitstempel im intern. Format als String zurueck"""
    import time 

    today = time.localtime()
        
    InternDatum = str(today.tm_year)+'-'+str(today.tm_mon)+'-'+str(today.tm_mday)
    InternZeit  = str(today.tm_hour)+':'+str(today.tm_min)+':'+str(today.tm_sec)
    InternDatumsformat = InternDatum + 'T' + InternZeit
    return InternDatumsformat

print(mystemp())