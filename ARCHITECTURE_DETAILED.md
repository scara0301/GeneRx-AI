# Personalized Drug AI - Detailed Architecture

## System Overview

```mermaid
graph TB
    subgraph Frontend["Frontend Layer"]
        UI[Streamlit UI]
        UI --> |User Input| Patient[Patient Information]
        UI --> |User Input| Genetic[Genetic Profile]
        UI --> |User Input| Drug[Drug Selection]
        UI --> |Display| Results[Results Visualization]
    end

    subgraph Backend["Backend Layer"]
        subgraph DataProcessing["Data Processing"]
            GP[Genetic Processor]
            DP[Drug Processor]
            PP[Patient Processor]
        end

        subgraph Models["ML Models"]
            GT[Genetic Transformer]
            DT[Drug Transformer]
            RP[Response Predictor]
        end

        subgraph Simulation["Simulation Engine"]
            EC[Efficacy Calculator]
            SE[Side Effects]
            RS[Response Simulator]
        end
    end

    Patient --> GP
    Genetic --> GP
    Drug --> DP
    GP --> GT
    DP --> DT
    GT --> RP
    DT --> RP
    RP --> EC
    RP --> SE
    EC --> RS
    SE --> RS
    RS --> Results
```

## Data Flow Architecture

```mermaid
sequenceDiagram
    participant User
    participant UI
    participant DataProcessor
    participant Model
    participant Simulator
    participant Visualizer

    User->>UI: Input Patient Data
    UI->>DataProcessor: Process Patient Info
    DataProcessor->>Model: Generate Predictions
    Model->>Simulator: Simulate Drug Response
    Simulator->>Visualizer: Create Visualizations
    Visualizer->>UI: Display Results
    UI->>User: Show Predictions
```

## Component Architecture

```mermaid
classDiagram
    class GeneticProcessor {
        +process_variants()
        +extract_features()
        +validate_data()
    }

    class DrugProcessor {
        +process_drug_info()
        +extract_molecular_features()
        +validate_drug_data()
    }

    class PatientProcessor {
        +process_patient_info()
        +combine_data()
        +validate_patient_data()
    }

    class GeneticTransformer {
        +encode_sequence()
        +process_variants()
        +generate_embeddings()
    }

    class DrugTransformer {
        +encode_molecule()
        +process_drug_info()
        +generate_embeddings()
    }

    class ResponsePredictor {
        +combine_features()
        +predict_efficacy()
        +predict_side_effects()
    }

    class Simulator {
        +calculate_efficacy()
        +simulate_response()
        +predict_side_effects()
    }

    GeneticProcessor --> GeneticTransformer
    DrugProcessor --> DrugTransformer
    PatientProcessor --> ResponsePredictor
    GeneticTransformer --> ResponsePredictor
    DrugTransformer --> ResponsePredictor
    ResponsePredictor --> Simulator
```

## System Components

### 1. Frontend Layer
```mermaid
graph LR
    subgraph UI["User Interface"]
        P[Patient Info]
        G[Genetic Profile]
        D[Drug Selection]
        R[Results]
    end
    
    P --> |Input| F[Form Handler]
    G --> |Input| F
    D --> |Input| F
    F --> |Process| B[Backend API]
    B --> |Response| R
```

### 2. Data Processing Layer
```mermaid
graph TD
    subgraph DP["Data Processing"]
        G[Genetic Data]
        D[Drug Data]
        P[Patient Data]
        
        G --> GP[Genetic Processor]
        D --> DP[Drug Processor]
        P --> PP[Patient Processor]
        
        GP --> FE[Feature Extraction]
        DP --> FE
        PP --> FE
        
        FE --> VD[Validation]
        VD --> PE[Preprocessing]
    end
```

### 3. Model Layer
```mermaid
graph TD
    subgraph ML["Machine Learning Models"]
        GT[Genetic Transformer]
        DT[Drug Transformer]
        RP[Response Predictor]
        
        GT --> |Embeddings| FM[Feature Merger]
        DT --> |Embeddings| FM
        FM --> RP
        
        RP --> |Predictions| EC[Efficacy Calculator]
        RP --> |Predictions| SE[Side Effects]
    end
```

### 4. Simulation Layer
```mermaid
graph LR
    subgraph SIM["Simulation Engine"]
        EC[Efficacy Calculator]
        SE[Side Effects]
        RS[Response Simulator]
        
        EC --> |Efficacy| RS
        SE --> |Side Effects| RS
        RS --> |Results| V[Visualizer]
    end
```

## Technical Stack

```mermaid
graph TD
    subgraph Frontend["Frontend Stack"]
        S[Streamlit]
        M[Matplotlib]
        N[NumPy]
    end
    
    subgraph Backend["Backend Stack"]
        P[PyTorch]
        T[Transformers]
        D[Pandas]
    end
    
    subgraph ML["ML Components"]
        BERT[BioBERT]
        TRANS[Transformer Models]
        SIM[Simulation Engine]
    end
    
    Frontend --> Backend
    Backend --> ML
```

## Data Flow Details

```mermaid
graph TD
    subgraph Input["Input Data"]
        P[Patient Info]
        G[Genetic Data]
        D[Drug Data]
    end
    
    subgraph Processing["Data Processing"]
        PP[Patient Processor]
        GP[Genetic Processor]
        DP[Drug Processor]
    end
    
    subgraph Model["ML Models"]
        GT[Genetic Transformer]
        DT[Drug Transformer]
        RP[Response Predictor]
    end
    
    subgraph Output["Output"]
        E[Efficacy]
        S[Side Effects]
        R[Response]
    end
    
    P --> PP
    G --> GP
    D --> DP
    
    PP --> RP
    GP --> GT
    DP --> DT
    
    GT --> RP
    DT --> RP
    
    RP --> E
    RP --> S
    RP --> R
```

## Deployment Architecture

```mermaid
graph TD
    subgraph Client["Client Layer"]
        B[Browser]
        M[Mobile App]
    end
    
    subgraph Server["Server Layer"]
        LB[Load Balancer]
        API[API Gateway]
        APP[Application Server]
        DB[(Database)]
    end
    
    subgraph ML["ML Services"]
        TM[Training Manager]
        PM[Prediction Manager]
        SM[Simulation Manager]
    end
    
    B --> LB
    M --> LB
    LB --> API
    API --> APP
    APP --> DB
    APP --> TM
    APP --> PM
    APP --> SM
``` 