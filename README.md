# AI-Agent

Python-baserad agent för att generera och validera svenska salongskonversationer från färdiga templates.

## Mål

Agenten ska kunna ta emot:

- färdiga conversation templates
- pooldata
- MLP-taxonomi
- språkprofil

och sedan:

1. bygga ett deterministiskt TurnPlan
2. generera kund- och AI-repliker
3. stämpla kundens repliker med MLP1 och optional MLP2
4. kontrollera hårda regler i Python
5. göra semantisk AI-validering
6. klassificera konversationen som:
   - SAFE
   - WEAK / TVEK
   - ERROR / FEL

## MLP-taxonomi

### MLP1

- Booking
- FAQ
- Tjänster
- SmallTalk
- Auth

### MLP2

- Booking → boka | omboka | avboka
- Tjänster → lägga till | ta bort | byta ut
- Auth → skapa | ändra
- FAQ → ingen MLP2
- SmallTalk → ingen MLP2

Generatorn får hela taxonomin men inget färdigt MLP-facit. Den genererar kundtexten och väljer därefter MLP-label utifrån vad kunden faktiskt uttrycker.

## Arkitektur

```text
ConversationTemplate
        +
PoolData
        +
MLPTaxonomy
        ↓
TurnPlanBuilder
        ↓
ConversationGenerator
        ↓
Dialogue + MLP labels
        ↓
DeterministicValidationPipeline
        ↓
SemanticTurnValidator
        ↓
SemanticDialogueValidator
        ↓
ConversationClassifier
        ↓
SAFE / WEAK / ERROR