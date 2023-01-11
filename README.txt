1. Zbiór danych testowych do problemu równoległego RPQ
2. O ile równoległy schrage powinien ulepszać rozwiązanie?
3. Kryterium wrzucania na daną listę uszeregowanych zadaniach po najmniejszym prep time?
    - jak to rozdzielić, zeby caly czas nie wrzucalo sie na jedna
4. Optymalizacja heurystyka AG
    - czy to powinno mieszac w kolejnosci dopiero po skonczeniu przez shrage?
    - czy moze na biezaco zmieniac rozwiazanie
5. Czy odrazu meta heurystyka czy zaczac od algorytmu schrage i wtedy poprawic metaheurystyka


AD1.
Teoretycznie kazał złączyć dwie listy i na tym testować, ale możemy to pominąć mysle
AD2.
Nie pytalem to by nic nie dalo

AD3.
Jedna lista N_g i z tego ma wrzucać

AD4. i AD5.
Podejscie takie że najpierw robimy Shrage z dwoma maszynami
A nastepnie od losowego rozwiazania algorytmem genetycznym chcemy uzyskac to co dal shrage lub lepiej
Dwa niezalezne podejscia i potem porownanie.

AD6. Jak wstawiac dane do algorytmu genetycznego
Niech liczby 1,2,3,4,5 ... n to numery zadań
Każde zadanie ma swoje prep time make time i delivery
liczba 0 oznacza przejscie na druga maszyne np ciąg
1 4 5 7 8 3 0 2 6 9
oznacza ze zadania 1,4,5,7,8,3 sa na maszynie 1
a zadania 2,4,6,9 na maszynie 2
ten otrzymany ciag wrzucamy do policzenia cmaxa i otrzymujemy wartość
nastepnie ta wartosc idzie do algorytmu genetycznego i on sobie mieli i wyznacza nowa kolejnosc liczb np:
9 6 5 7 8 3 0 2 1 4 i znowu to pójdzie na policzenie cmax i tak w kólko
Z perspektywy algorytmu genetycznego totalnie nie interesuje go co to znaczy
