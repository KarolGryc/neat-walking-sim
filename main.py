from Simulation import Simulation
import neat
import random

epoch = 0
SKIP_FIRST_EPOCHS = 50
DISPLAY_EVERY_EPOCH = 10

def eval_genomes(genomes, config):
    global epoch

    # Reset simulation once for the whole generation
    sim.reset()
    sim.make_walkers(len(genomes))
    
    # Create a neural network for each genome
    nets = []
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)

    NUM_ITERATIONS = int(min(1500, 300 + (epoch / 15) * 100))
    # Run the simulation for all walkers in parallel
    for _ in range(NUM_ITERATIONS):
        # Get inputs for all walkers
        inputs = [walker.info().as_array() for walker in sim.walkers]
        
        # Process inputs through each genome's network
        all_outputs = []
        for i, net in enumerate(nets):
            outputs = net.activate(inputs[i])
            all_outputs.append(outputs)
        
        sim.handle_events()
        if not sim.running:
            exit(1)

        # Update all walkers with their respective outputs
        sim.update(all_outputs)

        if epoch >= SKIP_FIRST_EPOCHS and epoch % DISPLAY_EVERY_EPOCH == 0:
            sim.draw()
    
    # Evaluate fitness for each genome
    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = sim.walkers[i].fitness()
        
    epoch += 1

if __name__ == "__main__":
    sim = Simulation()

    random.seed(1000)
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            'neat-config.ini')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(40))

    winner = p.run(eval_genomes, 200)

    # playback the best genome
    sim.reset()
    sim.make_walkers(1)
    best_genome = neat.nn.FeedForwardNetwork.create(winner, config)
    
    while sim.running:
        inputs = sim.walkers[0].info().as_array()
        outputs = best_genome.activate(inputs)
        sim.handle_events()
        if not sim.running:
            exit(1)
        
        sim.update([outputs])
        sim.draw()

        if sim.walkers[0].info().headAltitude < 0.4:
            print("Walker fell down, stopping simulation.")
            break
