from unittest import main, TestCase

import noregret as nr


class AbstractionTestCase(TestCase):
    KER = nr.FPKer()
    GAME = nr.OpenSpielGame(
        KER,
        (
            'turn_based_simultaneous_game('
            'game=goofspiel('
            'imp_info=True,'
            'num_cards=6,'
            'points_order=descending))'
        ),
    )
    GAME2 = nr.to_efg(KER, GAME)
    SEED_MOD = 2 ** 32
    K = 3
    DIVISOR = 2

    def seed(self, decision_point, _):
        return int.from_bytes(decision_point.encode()) % self.SEED_MOD

    def k(self, _, __):
        return self.K

    def embed(self, decision_point, action):
        np = self.KER.numpy
        dtype = self.KER.data_type
        element = (int(action.split()[-1]) - 1) // self.DIVISOR
        embedding = np.array([element], dtype)

        return embedding

    def cluster(self, embeddings, k):
        np = self.KER.numpy
        centroids = []

        for embedding in embeddings:
            for centroid in centroids:
                if np.allclose(embedding, centroid):
                    break
            else:
                centroids.append(embedding)

        return centroids

    def test_action_abstraction(self):
        abstraction = nr.RandomActionAbstraction(
            self.KER,
            self.GAME,
            self.seed,
            self.k,
        )
        game = nr.to_efg(self.KER, abstraction)
        R_row = nr.CFR_plus(self.KER, game.row_sequence_form_polytope)
        R_column = nr.CFR_plus(self.KER, game.column_sequence_form_polytope)
        x, y = nr.rm(
            game,
            R_row,
            R_column,
            alternation=True,
            prediction=True,
            progress_bar=False,
        )
        sigma = nr.SequenceFormStrategyProfile(
            self.KER,
            abstraction,
            (
                game.row_sequence_form_polytope.non_empty_sequences,
                game.column_sequence_form_polytope.non_empty_sequences,
            ),
            (x, y),
        )
        sigma = abstraction.to_lifted_behavioral_form(sigma)
        x2, y2 = self.GAME2.to_sequence_form(sigma)

        abstraction = nr.EmbeddingAbstraction(
            self.KER,
            self.GAME,
            self.embed,
            self.k,
            self.cluster,
        )
        game = nr.to_efg(self.KER, abstraction)
        R_row = nr.CFR_plus(self.KER, game.row_sequence_form_polytope)
        R_column = nr.CFR_plus(self.KER, game.column_sequence_form_polytope)
        x, y = nr.rm(
            game,
            R_row,
            R_column,
            alternation=True,
            prediction=True,
            progress_bar=False,
        )
        sigma = nr.SequenceFormStrategyProfile(
            self.KER,
            abstraction,
            (
                game.row_sequence_form_polytope.non_empty_sequences,
                game.column_sequence_form_polytope.non_empty_sequences,
            ),
            (x, y),
        )
        sigma = abstraction.to_lifted_behavioral_form(sigma)
        x3, y3 = self.GAME2.to_sequence_form(sigma)

        epsilon2 = self.GAME2.exploitability(x2, y2)
        epsilon3 = self.GAME2.exploitability(x3, y3)
        v2 = self.GAME2.expected_row_utility(x2, y3)
        v3 = -self.GAME2.expected_row_utility(x3, y2)
        v = (v2 + v3) / 2

        self.assertAlmostEqual(epsilon2, 1)
        self.assertAlmostEqual(epsilon3, 1)
        self.assertAlmostEqual(v, -1, 4)


if __name__ == '__main__':
    main()  # pragma: no cover
