#### 1. Articles

**Chollet**, en el seu article, explica que la *intel·ligència* no s'ha d'avaluar en àmbits concrets, ja que els sistemes poden estar influenciats per **"priors"**, que són els coneixements innats o preexistents del sistema.  
Ell proposa que la *intel·ligència* s'ha de mesurar com (eliminat) la facilitat del sistema a adquirir habilitats en nous àmbits.  
Partint d’aquesta idea, argumenta que el test de Turing no és útil per detectar o avaluar comportament intel·ligent. És fàcil perquè precisament aquest test avalua al sistema en un **àmbit concret** (el de l’imitar a un humà) i no evalua la capacitat del sistema per generalitzar o tractar àmbits fora del seu domini.  

Un exemple seria un sistema que fos portat amb coneixements naturals de forma reforçada portat de una base de coneixements prèvia increïblement gran, però que si li demana en programes en Python, retorna una resposta incorrecta ja que es troba fora del seu domini. Aquest sistema passaria el test de Turing però no seria intel·ligent segons Chollet i, segurament, no (passaria) trauria bona nota en el seu benchmark **ARC (abstraction reasoning corpus)**.

**Rodney Brooks** argumenta que els sistemes intel·ligents en el món real no han de ser centralitzats (com ChatGPT).  
Ell explica que com en els dos animals com els insectes, és millor aplicar un sistema descentralitzat, on sigui auditat per mòduls, on cada mòdul estigui connectat directament amb la realitat i organitzats per capes de prioritat.  

Amb aquesta distribució, s’evita crear un “model” del món en el sistema i el sistema passa a interactuar de forma intel·ligent d'una manera natural.

Donat aquest context, si partim d’aquesta lògica, ChatGPT és un sistema que té un únic model central que processa tots els inputs.  
Segons Brooks, això no permet que el sistema sigui intel·ligent perquè en primer lloc no pot processar tots els inputs del món real de forma parcel·lada. I segons ell, tindrà problemes com els **coll d’ampolla** que bloquegin el sistema en situacions complexes.  

Això, ChatGPT representa una versió model del món i, per tant, està limitat a aquest model, cosa que no el permet actuar de forma intel·ligent en el món real perquè aquest és més gran que aquest model, perquè hi ha infinites variacions que és impossible centralitzar i, per tant, hi ha aspectes del món real que estan fora del seu coneixement central.

#### 3. Mico i plàtan

Per resoldre aquest problema, hem d’aplicar estratègia de resolució.

1. Passem a FNC i després clàusules:

$$
\neg P(x, y, z, s) \lor P(z, y, z, camina(x, z, s))
$$

$$
\neg P(x, y, x, s) \lor P(y, y, y, porta(x, y, s))
$$

$$
\neg P(b, b, b, s) \lor arriba(puja(s))
$$

$$
P(a, b, c, s_1)
$$

Com volem saber si el mico arriba al plàtan, sabem que quan el mico puja a la cadira, arriba al plàtan. Per tant, fem refutació de:

$$
\neg P(b, b, b, s)
$$

això pregunta on està el mico amb sota el plàtan. Si trobem la clàusula {5}, sabem que és veritat. Un detall és el que el mico puja i per tant haurà arribat al plàtan.

**Clàusules:**

- $c_1$: $\{\neg P(x, y, z, s), P(z, y, z, camina(x, z, s))\}$
- $c_2$: $\{\neg P(x, y, x, s), P(y, y, y, porta(x, y, s))\}$
- $c_3$: $\{\neg P(b, b, b, s), arriba(puja(s))\}$
- $c_4$: $\{P(a, b, c, s_1)\}$
- $c_5$: $\{\neg P(b, b, b, s)\}$

---

Ara resolem per resolució per refutació:

$c_6 = c_5 + c_2$

$b/y$  
$s/porta(x, y, s)$

$$
= \{\neg P(y, y, y, porta(x, y, s))\} + c_2
$$

$$
= \{\neg P(x, y, z, s)\}
$$

---

$c_7 = c_6 + c_1$

$y/z$  
$s/camina(x, z, s)$

$$
= \{\neg P(z, y, z, camina(x, z, s))\} + c_1
$$

$$
= \{\neg P(x, y, z, s)\}
$$

---

$c_8 = c_7 + c_4$

$x/a$  
$y/b$
$z/c$  
$s/s_1$

$$
= \{\neg P(a, b, c, s_1)\} + c_4
$$

$$
= \{\}
$$

---

**clàusula buida!**


---

Hem arribat a una clàusula buida, això vol dir que amb la negació de $P(b, b, b, s)$ hem produït una inconsistència, cosa que vol dir que en cert, per tant el mico arriba sota el plàtan i l'hem verificat:

$$
P(b, b, b, s) \rightarrow arriba(puja(s))
$$

Sabem que el mico arriba. Hem comprovat la clàusula prèvia a la implicació.

Els passos que ha fet el mico per arribar són les implicacions fetes que es troben en color negre. I això ens ofereix la implicació que el mico arriba $arriba(puja(s))$, que ho ve després derivant al clàusula de l’objectiu.

#### 4. Description logics

1. 
$$
\text{PizzaBona} \sqsubseteq \neg \text{PizzaDolenta}
$$
(Tbox)
$$
\text{PizzaDolenta} \sqsubseteq \neg \text{PizzaBona}
$$
(això és que no pugui ser les dos coses a l’hora, es combinen)

$$
\text{Pizza} \equiv \text{PizzaBona} \sqcup \text{PizzaDolenta}
$$

2. 
$$
\text{PizzaBona} \sqsubseteq \exists \text{téBase.Cruixent} \sqcap \neg \exists \text{téTopping.Pinya}
$$
(Tbox)

3.
$$
\text{Base} \sqsubseteq \text{Cruixent} \sqcup \text{Tova}
$$
(Tbox)

4.
$$
\text{Pizza} \sqcap \text{Chicago} \sqsubseteq \text{Base} \sqcap \text{Tova}
$$
(Tbox)

5.
$$
\text{tipus(Pizza, Chicago)}
$$
(Abox)

Aquest Abox aquí per mi, implica:

$$
\text{tipus(A, B)}
$$

que el $A$ és de tipus $B$.

---

El lloc corresponent està indicat entre parèntesi:


(Abox) o (Tbox)

#### 5. Inducció

Per demostrar això, primer aplico l'algoritme d’eliminació de candidats i després analitzarem el resultat.

$G$ $\rightarrow$ general  
$S$ $\rightarrow$ específic

$$
G = \{\{\text{Covid}, ?\}, \{?, \text{Covid}\}\}
$$
$$
S = \{\{\text{Grip}, \text{Covid}\}\}
$$

$$
G = \{\{? \\, \text{Covid}\}\}
$$
$$
S = \{?,?\}
$$

$$
G = \{\}
$$
$$
S = \{??\}
$$

**No pot ser $?$** Perquè entraria un negatiu.

Per tant, **podem comprovar que no convergeixen**.  
Repetim: veiem que hi ha exemples contradictoris que fan que l'algoritme acabi amb conjunt buit.  
Per tant, indica que **no ha trobat una seqüència que li permet identificar un patró amb les regles donades**.  
Però **podem modificar la forma en què es mostren les dades** per trobar una convergència.

---

**(el que passa si canvien la forma de les dades)**

Tot i això, si canviem la forma en què estan representades les dades, **sí que podem trobar un patró amb aquest algoritme**.

Un exemple seria canviar la taula de forma que mostri una única entrada:

| Entrada A | Entrada B | Classe |
|-----------|-----------|--------|
| igual     | igual     | 0      |
| diferent  | diferent  | 1      |
| diferent  | diferent  | 1      |
| igual     | igual     | 0      |

on **diferent** significa que $EntradaA \neq EntradaB$,  
i **igual** que $EntradaA = EntradaB$.

Per tant, podem veure que:

$$
diferent = \{\{\text{Grip}, \text{Covid}\}, \{\text{Covid}, \text{Grip}\}\}
$$

$$
igual = \{\{\text{Grip}, \text{Grip}\}, \{\text{Covid}, \text{Covid}\}\}
$$

---

$$
G = \{?\}
$$
$$
S = \{\text{diferent}\}
$$

...

$$
G = \{\text{diferent}\}
$$
$$
S = \{\text{diferent}\}
$$

$\Rightarrow$ **Convergeix!** *(en aquest cas, modificant les dades)*

---

En aquest cas **sí que convergeix**.  
Per tant, hem demostrat que **si modifiquem la forma en què es mostren les dades**, podem aconseguir una convergència.  
Si això ho apliquem a les definicions anteriors, veiem que el concepte de la **clase 1** es pot representar així:

$$
\{\{\text{Grip}, \text{Covid}\}, \{\text{Covid}, \text{Grip}\}\}
$$

per tenir la **classe 1**,  
i la **classe 0**:

$$
\{\{\text{Grip}, \text{Grip}\}, \{\text{Covid}, \text{Covid}\}\}
$$
