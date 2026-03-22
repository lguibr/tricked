<script lang="ts">
    import { onMount } from "svelte";
    import { engine } from "$lib/state.svelte.js";

    let activeExpName = $state("");

    let experiments: Array<{name: string, config: any}> = $state([]);
    let expName = $state("new_experiment_v1");
    let dModel = $state(32);
    let numBlocks = $state(2);
    let simulations = $state(15);
    let unrollSteps = $state(3);
    let trainBatch = $state(256);
    let numGames = $state(200);
    let workers = $state(12);

    let isExistingRun = $state(false);
    let syncStatus = $state("Fetching configs...");

    const WANDB_URL = "http://localhost:8081/home";

    async function fetchConfigs() {
        try {
            const res = await fetch("http://127.0.0.1:8080/api/experiments");
            experiments = await res.json();
            if (experiments.length > 0 && expName === "new_experiment_v1") {
                applyConfig(experiments[0].name, experiments[0].config);
            }
        } catch(e) {
            console.error("Failed to fetch experiments", e);
        }
    }

    async function checkExperiment() {
        if (!expName) return;
        syncStatus = "Checking...";
        try {
            const res = await fetch(`http://127.0.0.1:8080/api/experiment/${expName}`);
            const data = await res.json();
            if (data.exists) {
                isExistingRun = true;
                syncStatus = "Experiment Found! Architecture Locked.";
                if (data.config.dModel) dModel = data.config.dModel;
                if (data.config.numBlocks) numBlocks = data.config.numBlocks;
            } else {
                isExistingRun = false;
                syncStatus = "New Experiment. Tabula Rasa.";
            }
        } catch (e) {
            syncStatus = "Backend unreachable.";
        }
    }

    function applyConfig(name: string, config: any) {
        expName = name;
        dModel = config.dModel || 32;
        numBlocks = config.numBlocks || 2;
        simulations = config.simulations || 15;
        unrollSteps = config.unrollSteps || 3;
        trainBatch = config.trainBatch || 256;
        numGames = config.numGames || 200;
        workers = config.workers || 12;
        checkExperiment();
    }

    function createNew() {
        expName = "custom_run_" + Math.floor(Math.random() * 1000);
        dModel = 64;
        numBlocks = 4;
        simulations = 40;
        trainBatch = 128;
        numGames = 500;
        workers = 16;
        checkExperiment();
    }

    async function deleteExperiment(name: string) {
        if (!confirm(`Are you sure you want to delete experiment ${name}? This will delete the runs/ directory permanently.`)) return;
        const res = await fetch(`http://127.0.0.1:8080/api/experiment/${name}`, { method: 'DELETE' });
        if (res.ok) {
            if (expName === name) createNew();
            await fetchConfigs();
            checkExperiment();
        }
    }

    async function startTraining() {
        if (!expName) return;
        const payload = {
            expName, dModel, numBlocks, simulations, unrollSteps, trainBatch, numGames, workers,
            tempDecaySteps: engine.tempDecaySteps,
            maxGumbelK: engine.maxGumbelK
        };
        const res = await fetch("http://127.0.0.1:8080/api/training/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        if (res.ok) {
            activeExpName = expName;
            syncStatus = "Execution in progress...";
            setTimeout(fetchConfigs, 2000); // refresh layout
        }
    }

    async function stopTraining() {
        await fetch("http://127.0.0.1:8080/api/training/stop", { method: "POST" });
    }

    $effect(() => {
        if (engine.isTraining && !activeExpName) {
             activeExpName = expName || "Unknown";
        }
    });

    onMount(() => {
        fetchConfigs();
        checkExperiment();
    });
</script>



<div class="bg-surface-dim font-body pb-8 relative w-full max-w-3xl mx-auto rounded-none mb-8 border border-outline-variant/20">
    <header class="w-full z-50 bg-surface flex items-center justify-between px-6 h-16 border border-outline-variant/20 border-t-0 border-x-0">
        <div class="flex items-center gap-3">
            <span class="material-symbols-outlined text-primary" style="font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;">terminal</span>
            <h1 class="text-xl font-bold tracking-tighter text-primary drop-shadow-[0_0_40px_var(--color-surface-tint)] font-headline uppercase leading-none mt-1">ORCHESTRATOR CONTROL</h1>
            {#if engine.isTraining}
                <div class="w-2 h-2 rounded-none bg-secondary drop-shadow-[0_0_40px_var(--color-surface-tint)] ml-2 mt-1 animate-pulse"></div>
            {/if}
        </div>
        <button onclick={fetchConfigs} class="text-on-surface-variant hover:text-primary transition-colors">
            <span class="material-symbols-outlined text-sm">refresh</span>
        </button>
    </header>

    <main class="pt-6 px-6 space-y-6">
        <!-- Profile Selector / Sync -->
        <section>
            <div class="flex justify-between items-end mb-4 px-1">
                <h2 class="font-headline text-on-surface-variant text-xs uppercase font-bold tracking-widest flex items-center gap-2">
                    Training Sessions
                </h2>
                <span class="text-[10px] uppercase font-bold tracking-widest {isExistingRun ? 'text-secondary' : 'text-primary'}">{syncStatus}</span>
            </div>
            
            <div class="flex gap-4 overflow-x-auto pb-4 no-scrollbar">
                
                <!-- create new button -->
                <!-- svelte-ignore a11y_click_events_have_key_events -->
                <!-- svelte-ignore a11y_no_static_element_interactions -->
                <div class="min-w-[120px] bg-surface flex flex-col items-center justify-center p-4 rounded-none cursor-pointer border border-dashed border-outline-variant hover:border-primary transition-all group" onclick={createNew}>
                    <span class="material-symbols-outlined text-on-surface-variant group-hover:text-primary mb-1">add</span>
                    <p class="font-headline text-[10px] uppercase tracking-widest font-bold text-on-surface-variant group-hover:text-primary">New Config</p>
                </div>

                {#each experiments as exp}
                    <!-- svelte-ignore a11y_click_events_have_key_events -->
                    <!-- svelte-ignore a11y_no_static_element_interactions -->
                    <div class="min-w-[200px] flex-shrink-0 relative bg-surface-container p-4 rounded-none transition-all cursor-pointer group {expName === exp.name ? 'border border-primary bg-surface-container-high' : 'border border-outline-variant/20 opacity-70 hover:opacity-100 hover:bg-surface-container-low'}" onclick={() => applyConfig(exp.name, exp.config)}>
                        <div class="flex justify-between items-start mb-2">
                            <span class="material-symbols-outlined {expName === exp.name ? 'text-primary' : 'text-on-surface-variant'} text-lg">memory</span>
                            <button onclick={(e) => { e.stopPropagation(); deleteExperiment(exp.name); }} class="opacity-0 group-hover:opacity-100 text-on-surface-variant hover:text-error transition-all" title="Delete experiment">
                                <span class="material-symbols-outlined text-[16px]">delete</span>
                            </button>
                        </div>
                        <p class="font-headline text-sm font-bold block text-on-surface truncate" title={exp.name}>{exp.name}</p>
                        <p class="text-[10px] text-on-surface-variant line-clamp-1 mt-1">D_{exp.config.dModel || '?'} • B_{exp.config.trainBatch || '?'}</p>
                        <div class="mt-3 h-1 w-full bg-surface-container-highest rounded-none overflow-hidden">
                            <div class="h-full {expName === exp.name ? 'w-full bg-primary drop-shadow-[0_0_40px_var(--color-surface-tint)]' : 'w-0'} transition-all duration-500"></div>
                        </div>
                    </div>
                {/each}
            </div>
        </section>

        <!-- Hyperparameter Inputs -->
        <section class="space-y-4">
            <div class="bg-surface-container p-5 rounded-none">
                <div class="flex items-center gap-2 mb-4">
                    <span class="material-symbols-outlined text-primary text-sm">tune</span>
                    <h3 class="font-headline text-xs font-bold uppercase tracking-widest mt-1 text-on-surface">Hyperparameter Tuning</h3>
                </div>
                
                <div class="grid grid-cols-2 gap-4">
                    <div class="space-y-1 col-span-2">
                        <label for="expName" class="text-[10px] text-on-surface-variant font-bold uppercase tracking-[0.05rem]">Experiment Name</label>
                        <div class="relative">
                            <input id="expName" type="text" bind:value={expName} onblur={checkExperiment} disabled={engine.isTraining} class="w-full bg-surface-container-highest border border-outline-variant/20 rounded-none text-sm text-primary font-headline py-2.5 px-3 outline-none transition-all disabled:opacity-50 focus:border-primary focus:bg-surface-container-high" />
                        </div>
                    </div>
                    
                    <div class="space-y-1">
                        <label for="dModel" class="text-[10px] text-on-surface-variant font-bold uppercase tracking-[0.05rem]">d_model</label>
                        <div class="relative">
                            <input id="dModel" type="number" bind:value={dModel} disabled={engine.isTraining || isExistingRun} class="w-full bg-surface-container-highest border border-outline-variant/20 rounded-none text-sm text-primary font-headline py-2.5 px-3 outline-none transition-all disabled:opacity-50 focus:border-primary focus:bg-surface-container-high" />
                        </div>
                    </div>
                    <div class="space-y-1">
                        <label for="numBlocks" class="text-[10px] text-on-surface-variant font-bold uppercase tracking-[0.05rem]">Depth (Blocks)</label>
                        <div class="relative">
                            <input id="numBlocks" type="number" bind:value={numBlocks} disabled={engine.isTraining || isExistingRun} class="w-full bg-surface-container-highest border border-outline-variant/20 rounded-none text-sm text-primary font-headline py-2.5 px-3 outline-none transition-all disabled:opacity-50 focus:border-primary focus:bg-surface-container-high" />
                        </div>
                    </div>
                    
                    <div class="space-y-1">
                        <label for="simulations" class="text-[10px] text-on-surface-variant font-bold uppercase tracking-[0.05rem]">MCTS Sims</label>
                        <div class="relative">
                            <input id="simulations" type="number" bind:value={simulations} disabled={engine.isTraining} class="w-full bg-surface-container-highest border border-outline-variant/20 rounded-none text-sm text-primary font-headline py-2.5 px-3 outline-none transition-all disabled:opacity-50 focus:border-primary focus:bg-surface-container-high" />
                        </div>
                    </div>
                    <div class="space-y-1">
                        <label for="trainBatch" class="text-[10px] text-on-surface-variant font-bold uppercase tracking-[0.05rem]">Batch Size</label>
                        <div class="relative">
                            <input id="trainBatch" type="number" bind:value={trainBatch} disabled={engine.isTraining} class="w-full bg-surface-container-highest border border-outline-variant/20 rounded-none text-sm text-primary font-headline py-2.5 px-3 outline-none transition-all disabled:opacity-50 focus:border-primary focus:bg-surface-container-high" />
                        </div>
                    </div>

                    <div class="space-y-1">
                        <label for="numGames" class="text-[10px] text-on-surface-variant font-bold uppercase tracking-[0.05rem]">Total Games</label>
                        <div class="relative">
                            <input id="numGames" type="number" bind:value={numGames} disabled={engine.isTraining} class="w-full bg-surface-container-highest border border-outline-variant/20 rounded-none text-sm text-primary font-headline py-2.5 px-3 outline-none transition-all disabled:opacity-50 focus:border-primary focus:bg-surface-container-high" />
                        </div>
                    </div>
                    <div class="space-y-1">
                        <label for="workers" class="text-[10px] text-on-surface-variant font-bold uppercase tracking-[0.05rem]">CPU Workers</label>
                        <div class="relative">
                            <input id="workers" type="number" bind:value={workers} disabled={engine.isTraining} class="w-full bg-surface-container-highest border border-outline-variant/20 rounded-none text-sm text-primary font-headline py-2.5 px-3 outline-none transition-all disabled:opacity-50 focus:border-primary focus:bg-surface-container-high" />
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- WandB Dashboard Button -->
        <a href={WANDB_URL} target="_blank" rel="noopener noreferrer" class="mb-4 w-full h-14 rounded-none bg-surface-container border border-outline-variant/20 hover:border-primary flex items-center justify-between px-6 group active:scale-[0.98] transition-all cursor-pointer hover:bg-surface-container-high block no-underline text-on-surface">
            <div class="flex items-center gap-3">
                <div class="w-8 h-8 rounded-none bg-surface-container-highest flex items-center justify-center border border-outline-variant">
                    <span class="material-symbols-outlined text-primary">monitoring</span>
                </div>
                <span class="font-headline text-sm font-bold tracking-wide mt-1">Weights & Biases Dashboard</span>
            </div>
            <span class="material-symbols-outlined text-on-surface-variant group-hover:translate-x-1 group-hover:text-primary transition-all">chevron_right</span>
        </a>

        <!-- Action Controls -->
        <div class="grid grid-cols-2 gap-4">
            <button onclick={startTraining} disabled={engine.isTraining || !expName} class="h-16 rounded-none bg-primary/10 border border-primary text-primary font-headline font-bold flex flex-col items-center justify-center hover:bg-primary/20 hover:drop-shadow-[0_0_40px_var(--color-surface-tint)] active:scale-95 transition-all disabled:opacity-50 disabled:cursor-not-allowed">
                <span class="material-symbols-outlined mb-1">play_arrow</span>
                <span class="text-[10px] uppercase tracking-widest mt-[-4px]">Initiate Training</span>
            </button>
            <button onclick={stopTraining} disabled={!engine.isTraining} class="h-16 rounded-none bg-error/10 border border-error text-error font-headline font-bold flex flex-col items-center justify-center hover:bg-error/20 hover:drop-shadow-[0_0_40px_var(--color-surface-tint)] active:scale-95 transition-all disabled:opacity-50 disabled:cursor-not-allowed">
                <span class="material-symbols-outlined mb-1">stop</span>
                <span class="text-[10px] uppercase tracking-widest mt-[-4px]">Halt Execution</span>
            </button>
        </div>
    </main>
</div>
