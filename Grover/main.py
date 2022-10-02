from manim import *

import numpy as np
import scipy.linalg as la

parameters = dict(
    n=4, steps=40, winner=0b1001
)

class GroverAnimation(Scene):
    def construct(self):
        # Set basic parameters
        n, winner = parameters['n'], parameters['winner']
        N = 2 ** n

        # Each superposition will get a dot for its amplitude on a separate line
        lines = VGroup(*(NumberLine(x_range=(-1, 1, .2), tick_size=.05, length=2, color=GRAY) for _ in range(N)))
        lines.arrange_in_grid(rows=2 ** (n//2), cols=2 ** (n//2), buff=MED_LARGE_BUFF)
        lines.shift(DOWN)

        template = f'{{:0>{n}b}}' # Formats in binary with padded zeros

        dots = [Dot(radius=.04, color=(RED if k == winner else WHITE)) for k in range(N)]
        labels = [Text(template.format(k), font_size=12, color=(RED if k == winner else WHITE)) for k in range(N)]

        for k, (dot, label, line) in enumerate(zip(dots, labels, lines)):
            label.next_to(line, direction=.75 * UP)
            dot.move_to(line.number_to_point(0))

        self.add(lines)
        self.play(*(FadeIn(label) for label in labels))
        self.play(*(FadeIn(dot) for dot in dots))

        # Preliminary: make the Grover matrix
        F = np.eye(N)
        F[winner, winner] = -1
        H = la.hadamard(N)
        D = -np.eye(N)
        D[0, 0] = 1

        # Normalize determinant
        G = (H @ D @ H @ F) / N

        dots = VGroup(*dots)

        ax = Axes(x_range=(0, parameters['steps']), y_range=(-1, 1, .2), tips=False, color=GRAY)
        ax.scale(.35)
        ax.shift(1.9 * UP)
        self.play(FadeIn(ax))

        #----Grover's algorithm----
        w = np.zeros(N)
        v = np.ones(N) / np.sqrt(N)

        q0 = Dot(ax.c2p(0, 1 / np.sqrt(N)), radius=.02, color=WHITE)
        q1 = Dot(ax.c2p(0, 1 / np.sqrt(N)), radius=.02, color=WHITE)
        for t in range(1, parameters['steps'] + 1):
            p0 = Dot(ax.c2p(t, float(v[N - 1 - winner])), radius=.02, color=WHITE)
            p1 = Dot(ax.c2p(t, float(v[winner])), radius=.02, color=RED)

            self.play(
                *(dot.animate.move_to(line.number_to_point(vk))
                  for vk, line, dot in zip(v, lines, dots)),
                FadeIn(p0), FadeIn(p1),
                run_time=.5
            )

            self.play(
                FadeIn(Line(q0.get_center(), p0.get_center(), color=WHITE, stroke_width=1)),
                FadeIn(Line(q1.get_center(), p1.get_center(), color=RED, stroke_width=1)),
                run_time=.25
            )

            w = v
            v = G @ w
            q0 = p0
            q1 = p1

        self.wait(1)


if __name__ == '__main__':
    pass
