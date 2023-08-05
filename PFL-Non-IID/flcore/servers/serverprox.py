from flcore.clients.clientprox import ClientProx
from flcore.servers.serverbase import Server


class FedProx(Server):
    def __init__(self, args, my_data_loader, times):
        super().__init__(args, my_data_loader, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, ClientProx)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.nodes}")
        print("Finished creating server and clients.")


    def train(self):
        for i in range(self.global_rounds + 1):
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0 and i != 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train(epoch=i)

            self.receive_models()
            self.aggregate_parameters()

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))

        self.save_results()
        self.save_global_model()
