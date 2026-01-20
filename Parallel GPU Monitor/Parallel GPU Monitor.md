# Parallel GPU Monitor - Architecture Diagrams

Here are accurate technical diagrams for the GPU monitoring system architecture and workflows:

## Diagram 1: System Architecture & Components

```mermaid
classDiagram
    class ParallelGPUMonitor {
        +start()
        +stop()
        +get_current_metrics()
        +get_stats()
        +get_history_snapshot()
        +plot_live()
        +print_live_terminal()
        +generate_report()
        -_monitoring_loop()
        -_get_gpu_metrics()
        -_save_statistics()
    }

    class StandaloneLiveMonitor {
        +run()
        -_signal_handler()
    }

    class DataStructures {
        <<Thread-Safe>>
        -metrics_queue: Queue(maxsize=1000)
        -history: dict
        -stats: dict
        -data_lock: threading.Lock()
    }

    class LoggingSystem {
        -csv_filehandle
        -csv_writer
        -json_file
        -log_dir
    }

    ParallelGPUMonitor o-- DataStructures : contains
    ParallelGPUMonitor o-- LoggingSystem : contains
    StandaloneLiveMonitor --> ParallelGPUMonitor : uses
    ParallelGPUMonitor --> threading.Thread : creates
    ParallelGPUMonitor --> subprocess : calls
```

## Diagram 2: Threading Model & Data Flow

```mermaid
flowchart TD
    A[Main Thread] --> B[ParallelGPUMonitor.start]
    B --> C[Create Thread with daemon=True]
    C --> D{monitoring flag = True}
    
    subgraph MonitorThread [Background Thread]
        E[_monitoring_loop]
        F[Interval Timer]
        G[Query nvidia-smi]
        H[Parse GPU Metrics]
        I[Update Data Structures]
        J[Write to Logs]
        
        E --> F
        F --> G
        G --> H
        H --> I
        I --> J
        J --> F
    end
    
    D --> E
    
    I --> K[Thread-Safe Queue]
    I --> L[History Arrays]
    I --> M[Statistics Dict]
    
    K --> N[Main Thread can pop]
    L --> O[Plotting Functions]
    M --> P[get_stats Method]
    
    A --> Q[Call stop]
    Q --> R{monitoring flag = False}
    R --> S[Thread joins timeout=5]
    S --> T[Save final statistics]
```

## Diagram 3: Data Structures & Thread Safety

```mermaid
flowchart LR
    subgraph MainThread [Main Thread Access]
        direction LR
        A[get_current_metrics] --> B[Queue.get_nowait]
        C[get_stats] --> D[data_lock.acquire]
        E[get_history_snapshot] --> D
        F[plot_live] --> D
    end
    
    subgraph MonitorThread [Monitor Thread Updates]
        direction LR
        G[Collect Metrics] --> H[Queue.put_nowait]
        I[Update History] --> J[data_lock.acquire]
        K[Update Stats] --> J
    end
    
    subgraph ThreadSafeStorage [Shared Data Structures]
        L[metrics_queue<br/>Queue maxsize=1000]
        M[history<br/>dict with arrays]
        N[stats<br/>dict with aggregates]
        
        L --> B
        L --> H
        
        M --> J
        N --> J
        
        M --> D
        N --> D
    end
    
    style MainThread fill:#e1f5fe
    style MonitorThread fill:#f3e5f5
    style ThreadSafeStorage fill:#e8f5e8
```

## Diagram 4: Monitoring Sequence & Lifecycle

```mermaid
sequenceDiagram
    participant Main as Main Thread
    participant Monitor as ParallelGPUMonitor
    participant Thread as Monitor Thread
    participant GPU as nvidia-smi
    participant Queue as Metrics Queue
    participant File as Log Files
    
    Main->>Monitor: __init__(update_interval=1)
    Monitor->>Monitor: Setup locks, queues, history
    
    Main->>Monitor: start()
    Monitor->>Thread: Create daemon thread
    Thread->>Thread: _monitoring_loop()
    
    loop Every update_interval
        Thread->>GPU: Query metrics
        GPU-->>Thread: Return GPU data
        Thread->>Queue: put_nowait(metrics)
        Thread->>Monitor: Update history/stats (with lock)
        Thread->>File: Write CSV line
    end
    
    Main->>Monitor: get_current_metrics()
    Monitor->>Queue: get_nowait()
    Queue-->>Main: Latest metrics
    
    Main->>Monitor: get_stats()
    Monitor->>Monitor: Acquire data_lock
    Monitor-->>Main: Copy of stats
    
    Main->>Monitor: plot_live()
    Monitor->>Monitor: Acquire data_lock
    Monitor-->>Main: Generate matplotlib plot
    
    Main->>Monitor: stop()
    Monitor->>Thread: Set monitoring=False
    Thread->>Thread: Exit loop
    Thread->>File: Close file handles
    Thread->>File: Save final statistics
    Thread-->>Main: Thread joins
```

## Diagram 5: Command Execution Flow

```mermaid
flowchart TD
    A[_monitoring_loop] --> B[Setup log files]
    B --> C[Start interval loop]
    
    C --> D[_get_gpu_metrics]
    D --> E[Build nvidia-smi command]
    E --> F[Execute subprocess]
    F --> G{Timeout or error?}
    
    G -- Success --> H[Parse CSV output]
    H --> I[Convert to float]
    I --> J[Return metrics dict]
    
    G -- Failure --> K[Log error]
    K --> L[Return None]
    
    J --> M{Valid metrics?}
    M -- Yes --> N[Process metrics]
    M -- No --> C
    
    N --> O[Update queue]
    N --> P[Update history with lock]
    N --> Q[Update stats with lock]
    N --> R[Write to log]
    
    O --> S{Queue full?}
    S -- Yes --> T[Remove oldest item]
    T --> U[Add new item]
    S -- No --> U
    
    R --> C
    
    style D fill:#e8f5e8
    style N fill:#e1f5fe
```
