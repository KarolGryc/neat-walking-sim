from Simulation import Simulation
import neat
import random
import pygame

epoch = 0
best_fitness = -float('inf')

SKIP_FIRST_EPOCHS = 50
DISPLAY_EVERY_EPOCH = 10

def eval_genomes(genomes, config):
    global epoch
    global best_fitness

    sim.reset()
    sim.make_walkers(len(genomes))
    
    nets = []
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)

    NUM_ITERATIONS = int(min(1500, 300 + (epoch / 15) * 100))
    for frame in range(NUM_ITERATIONS):
        inputs = sim.infos_array()
        
        all_outputs = []
        for i, net in enumerate(nets):
            outputs = net.activate(inputs[i])
            all_outputs.append(outputs)
        
        sim.handle_events()
        if not sim.running:
            exit(1)

        sim.update(all_outputs)

        if epoch >= SKIP_FIRST_EPOCHS and (epoch - SKIP_FIRST_EPOCHS) % DISPLAY_EVERY_EPOCH == 0:
            strings = [
                f"Epoch: {epoch}",
                f"Best fitness yet: {best_fitness:.2f}",
            ]
            sim.draw(strings)
        elif frame == 0:
            sim.screen.fill((64,64,64))
            font = pygame.font.Font(None, 36)
            text_surface = font.render(f"Epoch {epoch} in progress...", True, (192,192,192))
            sim.screen.blit(text_surface, (100, 100))
            pygame.display.flip()
        
    for i, (genome_id, genome) in enumerate(genomes):
        fitness = sim.walkers[i].fitness()
        genome.fitness = fitness
        if fitness > best_fitness:
            best_fitness = fitness
        
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

        info = sim.walkers[0].info()
        strings = [
            f"Distance traveled: {info.hDistance:.2f} m",
            f"Energy spent: {info.energySpent:.2f} J",
            f"Energy efficiency: {info.energySpent / info.hDistance:.2f} m/J",
        ]
        sim.draw()
