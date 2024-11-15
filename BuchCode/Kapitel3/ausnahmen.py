try:
    x = int(input("Bitte eine Ziffer eingeben: "))
except ValueError:
    print("Das war keine Ziffer")
finally:
    print('Es hat mich gefreut mit Ihnen zu interagieren...')