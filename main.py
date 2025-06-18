from Simulation import Simulation
from SimulationForParallel import SimulationForParallel
import neat
import random
import time
import multiprocessing as mp

best_fitness = -float('inf')
epoch = 0
SKIP_FIRST_EPOCHS = 500
DISPLAY_EVERY_EPOCH = 1

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
    global best_fitness
    # Reset simulation once for the whole generation
    sim.reset()
    sim.make_walkers(len(genomes))
    
    nets = []
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)

    # NUM_ITERATIONS = int(min(1500, 300 + (epoch / 15) * 100))
    NUM_ITERATIONS = 1500
    for frame in range(NUM_ITERATIONS):
        inputs = sim.infos_array()
        
        all_outputs = []
        for i, net in enumerate(nets):
            if (sim.walkers[i].is_dead()):
                continue
            outputs = net.activate(inputs[i])
            all_outputs.append(outputs)
        
        sim.handle_events()
        if not sim.running:
            exit(1)

        sim.update(all_outputs)

        if epoch % DISPLAY_EVERY_EPOCH == 0:
            sim.draw()
    
    # Evaluate fitness for each genome
    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = sim.walkers[i].fitness()

def playback_genome(simulation, genome):
    simulation.reset()
    simulation.make_walkers(1)
    best_genome = neat.nn.FeedForwardNetwork.create(winner, config)
    
    while simulation.running:
        inputs = simulation.walkers[0].info().as_array()
        outputs = best_genome.activate(inputs)
        simulation.handle_events()
        
        simulation.update([outputs])
        simulation.draw()

        if simulation.walkers[0].info().headAltitude < 0.4:
            time.sleep(3)
            print("Walker fell down, stopping simulation.")
            break

if __name__ == "__main__":
    global sim
    random.seed(1000)
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            'neat-config.ini')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(100))

    pe = neat.ParallelEvaluator(mp.cpu_count(), eval_genome)

    p.run(pe.evaluate, SKIP_FIRST_EPOCHS)
    sim = Simulation()
    winner = p.run(eval_genomes, 1000)

    # playback the best genome
    while True:
        sim.handle_events()
        if not sim.running:
            exit(1)

        best_genome = neat.nn.FeedForwardNetwork.create(winner, config)
        playback_genome(sim, best_genome)
