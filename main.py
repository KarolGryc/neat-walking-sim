from Simulation import Simulation
from SimulationForParallel import SimulationForParallel
import neat
import random
import time
import multiprocessing as mp
import visualize

EPOCHS_WITHOUT_RENDER = 1
EPOCHS_WITH_RENDER = 0

# Checkpoints
LOAD_FROM_CHECKPOINT = True
CHECKPOINT_RESTORE_FILE = 'test_results/test4/checkpoints/checkpoint-2999'
# CHECKPOINT_RESTORE_FILE = 'test_results/test8/checkpoints/checkpoint-2999'
CHECKPOINT_SAVE_FILE = 'checkpoints/checkpoint-'
CHECKPOINT_STEP = 20
SAVE_CHECKPOINTS = False

# Learning reports
REPORT_LEARNING_INFO = False
DRAW_RESULTS_GRAPHS = False

# Render results
DISPLAY_POPULATION_AT_FINISH = False
DISPLAY_WINNER_IN_LOOP = True

def eval_genome(genome, config):
    genome.fitness = 0.0
    sim = SimulationForParallel()
    sim.make_walker()
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    NUM_ITERATIONS = 1500
    for _ in range(NUM_ITERATIONS):
        if (sim.walker.is_dead()):
            break
        inputs = sim.walker.info().as_array()
        outputs = net.activate(inputs)
        sim.update(outputs)
    
    return sim.walker.fitness()

def eval_genomes(genomes, config):
    sim.reset()
    sim.make_walkers(len(genomes))
    
    nets = []
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)

    # NUM_ITERATIONS = int(min(1500, 300 + (epoch / 15) * 100))
    NUM_ITERATIONS = 1500
    for _ in range(NUM_ITERATIONS):
        inputs = sim.infos_array()
        
        all_outputs = []
        for i, net in enumerate(nets):
            outputs = net.activate(inputs[i])
            all_outputs.append(outputs)
        
        sim.handle_events()
        if not sim.running:
            exit(1)

        sim.update(all_outputs)
        sim.draw()
    
    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = sim.walkers[i].fitness()

def playback_genome(simulation, genome, playback_iterations=1500):
    simulation.reset()
    simulation.make_walkers(1)
    best_genome = neat.nn.FeedForwardNetwork.create(winner, config)
    

    total_angle = 0.0
    for _ in range(playback_iterations):
        inputs = simulation.walkers[0].info().as_array()
        outputs = best_genome.activate(inputs)
        simulation.handle_events()
        
        simulation.update([outputs])
        simulation.draw()

        simulation.handle_events()

        total_angle += simulation.walkers[0].info().torsoAngle
        if not sim.running:
            exit(0)


        if simulation.walkers[0].info().headAltitude < 0.4:
            time.sleep(3)
            print("Walker fell down, stopping simulation.")
            return
        
    print(f"Walked distance: {simulation.walkers[0].info().hDistance}\t Average angle: {(total_angle / playback_iterations) * 180 / 3.14159}")

if __name__ == "__main__":
    global sim
    random.seed(1000)
    
    winner = None
    stats = None

    # Prepare the population
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            'neat-config.ini')
    
    if LOAD_FROM_CHECKPOINT:
        population = neat.Checkpointer.restore_checkpoint(CHECKPOINT_RESTORE_FILE)
    else:
        population = neat.Population(config)
    
    # Add stats reporter to the population
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    if REPORT_LEARNING_INFO:
        out_reporter = neat.StdOutReporter(True)
        population.add_reporter(out_reporter)

    if SAVE_CHECKPOINTS:
        checkpointer = neat.Checkpointer(CHECKPOINT_STEP, filename_prefix=CHECKPOINT_SAVE_FILE)
        population.add_reporter(checkpointer)

    if EPOCHS_WITHOUT_RENDER > 0:
        pe = neat.ParallelEvaluator(mp.cpu_count(), eval_genome)
        winner = population.run(pe.evaluate, EPOCHS_WITHOUT_RENDER)

    if EPOCHS_WITH_RENDER > 0:
        sim = Simulation()
        winner = population.run(eval_genomes, EPOCHS_WITH_RENDER)

    if DRAW_RESULTS_GRAPHS:
        visualize.draw_net(config, winner, True, filename='best_network')
        visualize.plot_stats(stats, view=True, filename='fitness.svg', ylog=False)
        visualize.plot_species(stats, view=True, filename='speciation.svg')

    if DISPLAY_POPULATION_AT_FINISH:
        sim = Simulation()
        population.run(eval_genomes, 1)

    if DISPLAY_WINNER_IN_LOOP:
        sim = Simulation()
        if winner is None:
            print("No winner found!")
        else:
            while True:
                sim.handle_events()
                if not sim.running:
                    exit(0)

                best_genome = neat.nn.FeedForwardNetwork.create(winner, config)
                playback_genome(sim, best_genome)
