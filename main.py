from Simulation import Simulation
from WorldDrawer import WorldDrawer
import neat
import random
import pygame
import multiprocessing

epoch = 0
best_fitness = -float('inf')

SKIP_FIRST_EPOCHS = 100
DISPLAY_EVERY_EPOCH = 10
NUM_WORKERS = multiprocessing.cpu_count() - 1

sims = [Simulation() for _ in range(NUM_WORKERS)]
drawer = WorldDrawer()


def simulation_worker(simulation, networks, num_steps):
    for _ in range(num_steps):
        inputs = simulation.infos_array()
        all_outputs = [net.activate(inputs[i]) for i, net in enumerate(networks)]
        simulation.update(all_outputs)
    return [walker.fitness() for walker in simulation.walkers]



def eval_genomes(genomes, config):
    global epoch
    global best_fitness
    global sim
    global drawer

    # Create a neural network for each genome
    nets = []
    num_genomes = 0
    for genome_id, genome in genomes:
        genome.fitness = -1000
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        num_genomes += 1

    NUM_ITERATIONS = int(min(1500, 300 + (epoch / 15) * 100))

    # drawn epochs
    if epoch >= SKIP_FIRST_EPOCHS and (epoch - SKIP_FIRST_EPOCHS) % DISPLAY_EVERY_EPOCH == 0:

        sim = sims[0]
        sim.reset()
        sim.make_walkers(num_genomes)

        for _ in range(NUM_ITERATIONS):
            inputs = sim.infos_array()
            all_outputs = [net.activate(inputs[i]) for i, net in enumerate(nets)]
            
            drawer.handle_events()
            sim.update(all_outputs)

            strings = [
                f"Epoch: {epoch}",
                f"Best fitness yet: {best_fitness:.2f}",
            ]
            drawer.draw(strings)
        
        for i, (genome_id, genome) in enumerate(genomes):
            fitness = sim.walkers[i].fitness()
            genome.fitness = fitness
            if fitness > best_fitness:
                best_fitness = fitness

    else:
        per_sim_nets = [[]] * len(sims)
        
        for i, sim in enumerate(sims):
            per_sim_nets[i % len(sims)].append(net)
            sim.reset()
            sim.make_walkers(len(per_sim_nets[i]))

        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            results = pool.starmap(simulation_worker, [(sims[i], per_sim_nets[i], NUM_ITERATIONS) for i in range(NUM_WORKERS)])

            for i, (genome_id, genome) in enumerate(genomes):
                fitness = results[i // len(sims)][i % len(sims)]
                genome.fitness = fitness
                if fitness > best_fitness:
                    best_fitness = fitness
    


        
    epoch += 1


if __name__ == "__main__":
    random.seed(1000)
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            'neat-config.ini')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 200)

    drawer.sim = sims[0]
    sim.reset()
    sim.make_walkers(1)
    best_genome = neat.nn.FeedForwardNetwork.create(winner, config)
    
    while True:
        inputs = sim.walkers[0].info().as_array()
        outputs = best_genome.activate(inputs)
        drawer.handle_events()         
        sim.update([outputs])
        drawer.draw()