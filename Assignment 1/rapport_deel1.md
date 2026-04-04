# Quantitative Financial Risk Management
## Assignment 1: Portfolio VaR & ES (Deel 1)

---

## 1. Introductie en Methodologie

Het doel van deze initiële analyse is het meten en voorspellen van de Value-at-Risk (VaR) en Expected Shortfall (ES) op een 1-dag horizon voor een gediversifieerde portfolio. Deze portfolio bevat blootstelling aan aandelen, brede marktindices en rentestanden. Omdat we ook een Europees aandeel hebben opgenomen, is de portfolio tevens blootgesteld aan wisselkoersrisico (EUR/USD).

### 1.1 Data Acquisitie en Synchronisatie
Voor deze analyse is circa 10 jaar aan dagelijkse data verzameld (1 januari 2016 t/m 31 maart 2026) via Yahoo Finance. De financiële markten hebben op verschillende dagen gesloten beurzen (bijvoorbeeld wegens nationale feestdagen). Om deze ontbrekende observaties op te lossen, is de *forward-fill* (last-observation-carried-forward) methode toegepast. Dit is een logische en breed geaccepteerde aanname voor financiële risicomodellering: als een beurs gesloten is, blijft de prijsstelling gelijk aan de vorige handelsdag, en is het dagelijkse rendement simpelweg $0\%$. In dit sample zijn 5 ontbrekende waarden op deze manier gesynchroniseerd, resulterend in 2.667 handelsdagen.

### 1.2 Portfolio Samenstelling en Gewichten
De beginwaarde van de portfolio is vastgesteld op **$1.000.000**. De activa en hun bijbehorende, statische allocatiegewichten zijn als volgt gekozen:

*   **AAPL (25%) & MSFT (25%)**: Dit zijn twee dominante tech-aandelen en de grootste componenten van de markt. Een gelijke, aanzienlijke weging zorgt voor een large-cap growth focus.
*   **ASML.AS (20%)**: Dit Nederlandse aandeel is geïntroduceerd om directe blootstelling te creëren aan buitenlandse aandelen (Euronext Amsterdam). Het totale rendement voor de USD-portfolio is berekend als de som van het log-rendement in EUR en het log-rendement van de EUR/USD wisselkoers, $r_{total} = r_{eur} + r_{fx}$. Dit is een cruciale aanname voor de aggregatie van risico's binnen een USD-basis.
*   **^GSPC / S&P 500 (20%)**: Een brede Amerikaanse beursindex. Dit introduceert systematisch marktrisico en biedt diversificatie buiten de directe exposure van de specifieke tech-aandelen.
*   **^IRX (10%)**: Een 13-weken US Treasury Bill rente, welke fungeert als proxy voor een floating-rate lening (een Liability). Het lagere gewicht (10%) limiteert de volatiliteit tijdens renteschokken, maar biedt wel de benodigde rente-sensitiviteit. De P&L voor de lening wordt benaderd middels gemodificeerde duratie ($D_{mod} = 0.25$).

### 1.3 De "Rolling Window" Architectuur
Om de 'look-ahead bias' te elimineren en de rekenkracht van de hardware (M3 Pro chip) optimaal te benutten, is gekozen voor een geavanceerde **Rolling Window van 1.000 dagen**.
*   **Daily Re-estimation**: In plaats van een maandelijkse of kwartaal-update, worden de modelparameters ($\alpha, \beta, \omega$ voor GARCH en de vrijheidsgraden $\nu$ voor Student-t) **dagelijks opnieuw geschat**.
*   **Efficiency**: Dankzij geoptimaliseerde vectorized code kon het systeem deze ~1.667 volledige re-generaties in slechts circa **65 seconden** voltooien. Dit zorgt voor een uiterst actuele risico-inschatting die direct reageert op nieuwe markt-informatie.

![Returns Plot](/Users/joostvanasselt/.gemini/antigravity/brain/0e84eff7-488d-4f38-a1e6-7f414a2515ed/plots/02_returns.png)
*Figuur 1: Log-returns van de portfolio componenten, inclusief de rentewijzigingen van de T-Bill proxy. Bovenstaande plot illustreert duidelijk een sterke 'volatility clustering' in het eerste kwartaal van 2020 bij de aandelenkoersen.*

---

## 2. 1-Dag VaR en ES (Individueel en Portfolio)

Vijf verschillende kwantitatieve modellen zijn toegepast om het marktrisico te vangen: de passieve 'Variance-Covariance' modellen (Normale en Student-t verdelingen), een niet-parametrisch model (Historical Simulation), en de dynamische modellen (GARCH-CCC en Filtered Historical Simulation met EWMA).

### 2.1 Empirische Observaties
Voordat we de formele VaR getallen beoordelen, kijken we naar de correlatie tussen de empirische winst/verlies (Profit & Loss) en de theoretische Normale verdeling:

![Loss Distribution](/Users/joostvanasselt/.gemini/antigravity/brain/0e84eff7-488d-4f38-a1e6-7f414a2515ed/plots/04_loss_distribution.png)
*Figuur 2: Verliesdistributie en een QQ-Plot van de portfolio.*

In *Figuur 2* (linksonder: de QQ-Plot) is duidelijk te zien dat het portfolio rendement zogeheten *'fat tails'* heeft; de uiteinden van de werkelijke (empirische) verliezen duiken veel verder naar beneden/boven dan de rode 45-graden lijn suggereert. Hier trekken we een hele cruciale conclusie uit: modellen die uitsluitend uitgaan van een Normale verdeling zullen extreme negatieve marktschokken substantieel **onderschatten**.

### 2.3 Model-Specificaties en Optimalisatie
Geen enkel model is statisch gelaten; voor de Student-t verdeling zijn de vrijheidsgraden ($\nu$) per window dynamisch gefit met `scipy.stats.t.fit`. Hierdoor past de "fatness" van de staarten zich aan de historische context van de afgelopen 4 jaar aan.

![Student-t QQ-Plot](/Users/joostvanasselt/.gemini/antigravity/brain/0e84eff7-488d-4f38-a1e6-7f414a2515ed/plots/05_qq_student_t.png)
*Figuur 3: Vergelijking van de portfolio-rendementen tegenover een theoretische Student-t verdeling. De fit is aanzienlijk beter dan bij de Normale verdeling, wat de keuze voor een Student-t model rechtvaardigt.*

Voor het **GARCH-FHS model** is een technisch hoogstandje toegepast: om convergentieproblemen bij de optimalisatie te voorkomen, is de data per window geschaald naar een variantie van 1.0 en na de schatting weer teruggeschaald.
*   **Logic**: FHS werkt door de gestandaardiseerde residuen ($z_t = \epsilon_t / \sigma_t$) uit de GARCH-fit te gebruiken om de historische distributie te herstellen. Het risico-kwantiel wordt bepaald op basis van deze $z_t$-reeks en vervolgens vermenigvuldigd met de voorspelde voorwaardelijke volatiliteit ($\sigma_{t+1}$) voor de volgende dag.

### 2.2 VaR (99%) & ES (97.5%) Waarden
Onderstaande tabel toont een momentopname van het voorspelde absolute dollarverlies voor een enkele dag, gebaseerd op de gehele data-periode. ES verwacht hogere waarden, omdat dit conditioneel rekent mét het 'tail risk' (het verwachte verlies, gegeven dát wij we VaR grens überhaupt passeren). Om die reden evalueren wij Normaliter de ES op 97.5% voor coherentie met een 99% VaR. 

| Instrument | Normal VaR (99%) | Student-t VaR (99%) | Historical VaR (99%) |
| :--- | :--- | :--- | :--- |
| **AAPL** | $10,191 | $11,264 | $12,316 |
| **MSFT** | $9,508 | $10,507 | $11,052 |
| **ASML.AS** (inc. FX) | $9,783 | $10,812 | $11,502 |
| **^GSPC** | $5,119 | $5,656 | $6,635 |
| **Portfolio (Totaal)** | **$27,658** | **$30,579** | **$35,367** |

De tabel bevestigt onze aannames uit de QQ-Plot: De "Student-t" verdeling en "Historical Simulation" rapporteren beide een fors hogere absolute VaR ($30.579 en $35.367 voor de totale portfolio) ten opzichte van de Normale verdeling. Historisch simuleren is extreem conservatief omdat de extreme marktcrashes van 2020 de dataset zwaar beïnvloeden.

### 2.4 Theoretische Onderbouwing van de Modellen
Om de 1-dag VaR en ES te berekenen, zijn vijf fundamentaal verschillende benaderingen gebruikt:

1.  **Parametrisch: Normale Variance-Covariance**
    *   *Logica*: Dit model gaat uit van een multivariate normale verdeling. Het portfoliorisico wordt berekend als $\sigma_p = \sqrt{w^T \Sigma w}$, waarbij $\Sigma$ de covariantie-matrix is. De VaR is simpelweg $\mu_p - z_{\alpha} \sigma_p$.
    *   *Beperking*: Het negeert 'fat tails' en gaat ervan uit dat extreme beurscrashes nagenoeg onmogelijk zijn.

2.  **Parametrisch: Student-t Variance-Covariance**
    *   *Logica*: Vergelijkbaar met het normale model, maar de verdeling heeft dikkere staarten (fat tails). De vorm wordt bepaald door de vrijheidsgraden ($\nu$). Hoe lager $\nu$, hoe groter de kans op extreme uitschieters. In dit project wordt $\nu$ dagelijks geoptimaliseerd.

3.  **Non-Parametrisch: Historische Simulatie (HS)**
    *   *Logica*: Dit model maakt geen enkele aanname over de vorm van de verdeling. Het neemt simpelweg de werkelijke rendementen van de afgelopen 1.000 dagen en kijkt wat het 1% slechtste resultaat was.
    *   *Voordeel*: Het vangt automatisch alle historische correlaties en extreme gebeurtenissen op.

4.  **Semi-Parametrisch: EWMA (Exponentially Weighted Moving Average)**
    *   *Logica*: Volatiliteit is niet constant. EWMA reageert snel op nieuwe paniek in de markt door recente rendementen zwaarder te laten wegen ($\lambda = 0.94$). De VaR "ademt" mee met de dagelijkse onrust.

5.  **Geavanceerd: GARCH(1,1) met Filtered Historical Simulation (FHS)**
    *   *Logica*: Dit is een "best-of-both-worlds" aanpak. Eerst filtert een GARCH-model de huidige volatiliteits-ruis uit de data. Daarna wordt op de 'schone' residuen een historische simulatie losgelaten. Dit model kan zowel snel reageren op nieuwe volatiliteit (GARCH) als extreme staartrisico's uit het verleden onthouden (FHS).

---

## 3. Backtesting en Evaluatie

Een VaR-model is pas valide als het daadwerkelijk standhoudt wanneer we het in het verleden toetsen tegenover out-of-sample data (Backtesting). We kijken specifiek naar het aantal 'schendingen' (Violations): het aantal dagen waarop de portfolio *meer* geld verloor dan het specifieke model voorspelde. 

Voor een observatieperiode van deze grootte ($\pm$1.667 werkdagen backtested out-of-sample), is het *verwachte aantal violations* voor the 99% VaR **16.67**.

![Backtesting Violations](/Users/joostvanasselt/.gemini/antigravity/brain/0e84eff7-488d-4f38-a1e6-7f414a2515ed/plots/loss_vs_var_es.png)
*Figuur 3: Buiten-sample Portfolio Loss (grijze lijn) tegenover geselecteerde VaR & ES Modellen. De rode stippen geven Violation-dagen voor het FHS model aan.*

### 3.1 Model Prestaties & Interpretatie

**Tabel 1: VaR-Prestaties (99% Betrouwbaarheid)**
| Model | Werkelijke Schendingen | VaR P-waarde (Kupiec) | Status |
| :--- | :---: | :---: | :--- |
| **Normal V-C** | 38 | 0.000 | Afgekeurd |
| **Student-t V-C** | 24 | 0.071 | Geaccepteerd |
| **EWMA** | 37 | 0.000 | Afgekeurd |
| **Historical Simulation** | 16 | 0.869 | Geaccepteerd |
| **GARCH-FHS** | 22 | 0.189 | Geaccepteerd |

**Tabel 2: Shortfall Analyse (Expected ES vs. Actual Loss)**
Een cruciale eis van de opdracht is de vergelijking tussen het *voorspelde* verlies (ES) en het *werkelijke* gemiddelde verlies op dagen van een overschrijding.

| Model | Voorspeld Gem. ES | Werkelijk Gem. Verlies | Onderschatting | ES P-waarde |
| :--- | :---: | :---: | :---: | :---: |
| **Normal V-C** | $27.846 | $40.668 | -31.5% | 0.0001 |
| **Student-t V-C** | $36.375 | $46.019 | -21.0% | 0.0134 |
| **EWMA** | $27.307 | $35.217 | -22.5% | 0.0000 |
| **Historical Simulation** | $37.217 | $52.228 | -28.7% | 0.0079 |
| **GARCH-FHS** | $29.176 | $35.010 | -16.7% | 0.0005 |

*Opmerking: De ES p-waarde is berekend middels de 'Rescaled Residuals' t-test (McNeil & Frey).*

![Spacing Test](/Users/joostvanasselt/.gemini/antigravity/brain/0e84eff7-488d-4f38-a1e6-7f414a2515ed/plots/spacings_qq_plots.png)
*Figuur 5: Spacing Test QQ-plots. Deze grafiek toont de tijdsintervallen tussen overschrijdingen aan tegenover een theoretische exponentiële verdeling. Voor de geaccepteerde modellen (zoals FHS) volgen de punten de vloeibare lijn, wat duidt op onafhankelijke schendingen zonder risk-clustering.*

**Intuitieve Conclusies en Kritische Reflectie:**

1.  **De Fat-Tail Valkuil:** Statische modellen gebaseerd op een normale verdeling overschrijden spectaculair hun eigen risicolimieten.
2.  **Het "Eerlijke" verhaal over de Expected Shortfall:** Uit *Tabel 2* blijkt dat hoewel modellen als GARCH-FHS de *frequentie* van crashes (VaR) goed voorspellen, zelfs de meest geavanceerde modellen de absolute *diepte* van de verliezen systematisch onderschatten (met -16.7% tot -31.5%). Dit is een kritische ontdekking: tijdens de COVID-schok van maart 2020 waren de verliezen grilliger dan de theorie voorspelde ($p < 0.05$ voor alle modellen op ES). Dit toont aan dat in tijden van extreme crisis de marktdynamiek sneller verandert dan een her-schattend model kan bijhouden.
3.  **Onafhankelijkheid (Spacing Test):** Naast de frequentie is ook de timing van overschrijdingen getoetst. Middels een **Spacing Test** (waarbij de tijd tussen overschrijdingen is vergeleken met een exponentiële verdeling) hebben we gecontroleerd op clustering. Voor FHS kon de hypothese van onafhankelijkheid niet worden verworpen, wat betekent dat de verliezen zich niet onnatuurlijk 'opstapelen' – een teken van een gezond risicomodel.
4.  **De Meest Gebalanceerde Keuze:** In *Figuur 4* fluctueert de GARCH FHS lijn vloeiend mee. Wanneer er een beursexplosie plaatsvindt, trekt GARCH bliksemsnel de VaR-muren omhoog. Zodra het stof inzinkt, zakt de VaR progressief in elkaar. GARCH FHS overleeft statistisch gezien de VaR-validatie ($p=0.189$).

We kunnen op basis van deze eerste analyse vaststellen dat de statische VaR modellen het reële financiële risico van dit aandelen- en FX-mandje systematisch onderwaarderen, en dat dynamische non-parametrische schaling (FHS) momenteel het veiligst is.
