# plot.py

from abc import ABC
from typing import Any, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
# from typeguard import typechecked
from csd.typings.global_result import GlobalResult
from csd.typings.typing import ResultExecution
from csd.utils.util import set_current_time, _fix_path, set_friendly_time
import numpy as np
# from csd.config import logger


class Plot(ABC):
    """ Class for plotting the results

    """

    # @typechecked
    def __init__(self, path: str, alphas: List[float] = None, number_modes: int = 1):
        if alphas is None:
            raise ValueError("alphas not set.")
        self._alphas = alphas
        self._path = path

    def success_probabilities_all_alphas(self,
                                         number_modes: List[int],
                                         number_ancillas: List[int],
                                         global_results: List[GlobalResult],
                                         save_plot: Optional[bool] = False,
                                         apply_log: Optional[bool] = False,
                                         squeezing: Optional[bool] = True,
                                         non_squeezing: Optional[bool] = False,
                                         plot_ancillas: Optional[bool] = False,
                                         interactive_plot: Optional[bool] = False,
                                         best_codebook: Optional[bool] = False) -> None:
        if best_codebook is None:
            best_codebook = False
        fig = plt.figure(figsize=(25, 20))
        suptitle_prefix = "Average" if not best_codebook else 'Best Codebook'
        apply_log_prefix = "" if not best_codebook else 'Best Codebook '
        fig.suptitle(f"{suptitle_prefix} Success Probability" if not apply_log
                     else f"{apply_log_prefix}Success Probability decreasing rate",
                     fontsize=20)

        for idx, one_alpha in enumerate(self._alphas):
            homodyne_probabilities: List[float] = []
            helstrom_probabilities: List[float] = []
            squeezed_probabilities = []
            non_squeezed_probabilities = []

            probs_labels = []
            for number_mode in number_modes:
                helstrom_probability, homodyne_probability = self._compute_hels_homo_prob(
                    alpha=one_alpha,
                    apply_log=apply_log if apply_log is not None else False,
                    number_mode=number_mode,
                    global_results=global_results,
                    best_codebook=best_codebook)
                helstrom_probabilities.append(helstrom_probability)
                homodyne_probabilities.append(homodyne_probability)

                squeezed_probabilities_mode_i = []
                non_squeezed_probabilities_mode_i = []

                for ancilla_i in number_ancillas:
                    squeezed_probability_ancilla_i = [global_result.success_probability
                                                      if not best_codebook else global_result.best_success_probability
                                                      for global_result in global_results
                                                      if (global_result.number_modes == number_mode and
                                                          global_result.number_ancillas == ancilla_i and
                                                          global_result.squeezing and
                                                          global_result.alpha == one_alpha)]
                    non_squeezed_probability_ancilla_i = [global_result.success_probability
                                                          if not best_codebook
                                                          else global_result.best_success_probability
                                                          for global_result in global_results
                                                          if (global_result.number_modes == number_mode and
                                                              global_result.number_ancillas == ancilla_i and
                                                              not global_result.squeezing and
                                                              global_result.alpha == one_alpha)]
                    if len(squeezed_probability_ancilla_i) > 1:
                        raise ValueError("more than one squeezed_probability found!")
                    if len(non_squeezed_probability_ancilla_i) > 1:
                        raise ValueError("more than one non_squeezed_probability found!")
                    if len(squeezed_probability_ancilla_i) == 0:
                        squeezed_probability_ancilla_i.append(0.0)
                    squeezed_probabilities_mode_i.append(squeezed_probability_ancilla_i.pop(0))
                    if len(non_squeezed_probability_ancilla_i) == 0:
                        non_squeezed_probability_ancilla_i.append(0.0)
                    non_squeezed_probabilities_mode_i.append(non_squeezed_probability_ancilla_i.pop(0))
                squeezed_probabilities.append(squeezed_probabilities_mode_i)
                non_squeezed_probabilities.append(non_squeezed_probabilities_mode_i)

            for ancilla_i in number_ancillas:
                if plot_ancillas or (not plot_ancillas and ancilla_i == 0):
                    sq_prob_ancilla_i = [sq_prob.pop(0) for sq_prob in squeezed_probabilities]
                    if squeezing:
                        probs_labels.append((sq_prob_ancilla_i, f"pSucc Squeez anc:{ancilla_i}"))
                    non_sq_prob_ancilla_i = [non_sq_prob.pop(0) for non_sq_prob in non_squeezed_probabilities]
                    if non_squeezing:
                        probs_labels.append((non_sq_prob_ancilla_i, f"pSucc No Squeez anc:{ancilla_i}"))
            probs_labels.append((homodyne_probabilities, "pSucc Homodyne"))
            probs_labels.append((helstrom_probabilities, "pSucc Helstrom"))

            ax = fig.add_subplot(4, 4, idx + 1 % 4)
            # ax.set_ylim([0, 1]) if not apply_log else ax.set_ylim([-1, 0])
            ax.set_title(f"$\\alpha$={np.round(one_alpha, 2)}", fontsize=14)

            lines = self._plot_lines_with_appropiate_colors(number_modes, probs_labels)
            new_lines = lines.copy()
            plt_lines = []
            for line in new_lines:
                plt_line, = ax.plot(line[0], line[1], label=line[2], color=line[3], linestyle=line[4])
                plt_lines.append(plt_line)

            set_ylabel_prefix = "Average" if not best_codebook else 'Best Codebook'
            set_ylabel_apply_log_prefix = "" if not best_codebook else 'Best Codebook '
            ax.set_xticks(number_modes)
            ax.legend(facecolor='silver', framealpha=0.7)
            ax.set_xlabel('number modes')
            ax.set_ylabel(
                f'{set_ylabel_prefix} Success Probabilities'
                if not apply_log else f'{set_ylabel_apply_log_prefix}Success Probability decreasing rate')
            ax.patch.set_facecolor('silver')
            # ax.patch.set_alpha(0.7)
        plt.subplots_adjust(hspace=0.4)
        fig.patch.set_facecolor('lightgrey')
        # fig.patch.set_alpha(0.7)
        prefix_suffix = 'best_' if best_codebook else ''
        suffix = f"_{prefix_suffix}probs_all" if not apply_log else f"_logs_{prefix_suffix}probs_all"
        self._show_or_save_plot(save_plot=save_plot if save_plot is not None else False,
                                suffix=suffix, fig=fig)

    def _compute_hels_homo_probs(self,
                                 apply_log: bool,
                                 number_mode: int,
                                 global_results: List[GlobalResult],
                                 alphas: List[float],
                                 best_codebook: Optional[bool] = False) -> Tuple[List[float], List[float]]:
        if best_codebook is None:
            best_codebook = False
        helstrom_probabilities: List[float] = []
        homodyne_probabilities: List[float] = []

        for alpha in alphas:
            helstrom_probability, homodyne_probability = self._compute_hels_homo_prob(alpha=alpha,
                                                                                      apply_log=apply_log,
                                                                                      number_mode=number_mode,
                                                                                      global_results=global_results,
                                                                                      best_codebook=best_codebook)
            helstrom_probabilities.append(helstrom_probability)
            homodyne_probabilities.append(homodyne_probability)
        return helstrom_probabilities, homodyne_probabilities

    def _compute_hels_homo_prob(self,
                                alpha: float,
                                apply_log: bool,
                                number_mode: int,
                                global_results: List[GlobalResult],
                                best_codebook: bool) -> Tuple[float, float]:

        homodyne_probability_all = [global_result.homodyne_probability
                                    if not best_codebook else global_result.best_homodyne_probability
                                    for global_result in global_results
                                    if (global_result.number_modes == number_mode and
                                        global_result.alpha == alpha)]
        homodyne_probability = homodyne_probability_all.pop() if len(homodyne_probability_all) > 0 else 0.0
        helstrom_probability_all = [global_result.helstrom_probability
                                    if not best_codebook else global_result.best_helstrom_probability
                                    for global_result in global_results
                                    if (global_result.number_modes == number_mode and
                                        global_result.alpha == alpha and
                                        global_result.squeezing)]
        helstrom_probability = helstrom_probability_all.pop() if len(helstrom_probability_all) > 0 else 0.0

        if apply_log and homodyne_probability > 0.0:
            homodyne_probability = np.log(homodyne_probability)
        if apply_log and helstrom_probability > 0.0:
            helstrom_probability = np.log(helstrom_probability)

        return helstrom_probability, homodyne_probability

    def _plot_lines_with_appropiate_colors(self,
                                           number_modes: List[int],
                                           probs_labels: List[Tuple[List[float], str]]) -> List[List[object]]:

        lines = []
        for prob, label in probs_labels:
            if label.find('Homodyne') != -1:
                color = 'black'
                linestyle = 'solid'
            if label.find('Helstrom') != -1:
                color = 'dimgrey'
                linestyle = 'solid'
            if label.find('pSucc Squeez') != -1:
                linestyle = 'solid'
                if label.find('anc:0') != -1:
                    color = 'red'
                if label.find('anc:1') != -1:
                    color = 'darkorange'
                if label.find('anc:2') != -1:
                    color = 'gold'
                if label.find('anc:3') != -1:
                    color = 'yellowgreen'
            if label.find('pSucc No Squeez') != -1:
                linestyle = 'dashed'
                if label.find('anc:0') != -1:
                    color = 'red'
                if label.find('anc:1') != -1:
                    color = 'darkorange'
                if label.find('anc:2') != -1:
                    color = 'gold'
                if label.find('anc:3') != -1:
                    color = 'yellowgreen'
            line = [number_modes, prob, label, color, linestyle]
            lines.append(line)
        return lines

    def success_probabilities_one_alpha(self,
                                        one_alpha: float,
                                        number_modes: List[int],
                                        number_ancillas: List[int],
                                        global_results: List[GlobalResult],
                                        save_plot: Optional[bool] = False,
                                        apply_log: Optional[bool] = False,
                                        interactive_plot: Optional[bool] = False,
                                        best_codebook: Optional[bool] = False) -> None:
        if best_codebook is None:
            best_codebook = False
        homodyne_probabilities: List[float] = []
        helstrom_probabilities: List[float] = []
        squeezed_probabilities = []
        non_squeezed_probabilities = []

        probs_labels = []
        for number_mode in number_modes:
            helstrom_probability, homodyne_probability = self._compute_hels_homo_prob(
                alpha=one_alpha,
                apply_log=apply_log if apply_log is not None else False,
                number_mode=number_mode,
                global_results=global_results,
                best_codebook=best_codebook)
            helstrom_probabilities.append(helstrom_probability)
            homodyne_probabilities.append(homodyne_probability)

            squeezed_probabilities_mode_i = []
            non_squeezed_probabilities_mode_i = []
            for ancilla_i in number_ancillas:
                squeezed_probability_ancilla_i = [global_result.success_probability
                                                  if not best_codebook else global_result.best_success_probability
                                                  for global_result in global_results
                                                  if (global_result.number_modes == number_mode and
                                                      global_result.number_ancillas == ancilla_i and
                                                      global_result.squeezing and
                                                      global_result.alpha == one_alpha)]
                non_squeezed_probability_ancilla_i = [global_result.success_probability
                                                      if not best_codebook else global_result.best_success_probability
                                                      for global_result in global_results
                                                      if (global_result.number_modes == number_mode and
                                                          global_result.number_ancillas == ancilla_i and
                                                          not global_result.squeezing and
                                                          global_result.alpha == one_alpha)]
                if len(squeezed_probability_ancilla_i) > 1:
                    raise ValueError("more than one squeezed_probability found!")
                if len(non_squeezed_probability_ancilla_i) > 1:
                    raise ValueError("more than one non_squeezed_probability found!")
                if len(squeezed_probability_ancilla_i) == 0:
                    squeezed_probability_ancilla_i.append(0.0)
                squeezed_probabilities_mode_i.append(squeezed_probability_ancilla_i.pop(0))
                if len(non_squeezed_probability_ancilla_i) == 0:
                    non_squeezed_probability_ancilla_i.append(0.0)
                non_squeezed_probabilities_mode_i.append(non_squeezed_probability_ancilla_i.pop(0))
            squeezed_probabilities.append(squeezed_probabilities_mode_i)
            non_squeezed_probabilities.append(non_squeezed_probabilities_mode_i)

        for ancilla_i in number_ancillas:
            sq_prob_ancilla_i = [sq_prob.pop(0) for sq_prob in squeezed_probabilities]
            probs_labels.append((sq_prob_ancilla_i, f"pSucc Squeez anc:{ancilla_i}"))
            non_sq_prob_ancilla_i = [non_sq_prob.pop(0) for non_sq_prob in non_squeezed_probabilities]
            probs_labels.append((non_sq_prob_ancilla_i, f"pSucc No Squeez anc:{ancilla_i}"))
        probs_labels.append((homodyne_probabilities, "pSucc Homodyne"))
        probs_labels.append((helstrom_probabilities, "pSucc Helstrom"))
        title_prefix = "Average" if not best_codebook else 'Best Codebook'
        apply_log_title_prefix = '' if not best_codebook else 'Best Codebook '
        title = (f"{title_prefix} Success Probability for $\\alpha$={np.round(one_alpha, 2)}"
                 if not apply_log else
                 f"{apply_log_title_prefix}Success Probability decreasing rate for $\\alpha$={np.round(one_alpha, 2)}")
        # fig, axes = plt.subplots(figsize=[10, 8])
        # plt.title(title, fontsize=20)
        # fig.patch.set_facecolor('lightgrey')
        # fig.patch.set_alpha(0.7)
        # axes.patch.set_facecolor('silver')
        # axes.patch.set_alpha(0.7)

        # lines = self._plot_lines_with_appropiate_colors(number_modes, probs_labels, axes)

        # axes.set_xticks(number_modes)
        # # axes.set_ylim([0, 1]) if not apply_log else axes.set_ylim([-1, 0])
        # plt.legend(facecolor='silver', framealpha=0.7)
        # plt.xlabel('number modes')
        # plt.ylabel('Average Success Probabilities' if not apply_log else 'Success Probability decreasing rate')

        # self._show_or_save_plot(save_plot=save_plot if save_plot is not None else False,
        #                         suffix=suffix, fig=fig)
        prefix_suffix = 'best_' if best_codebook else ''
        suffix = (f"_{prefix_suffix}probs_{str(np.round(one_alpha, 2))}"
                  if not apply_log else f"_logs_{prefix_suffix}probs_{str(np.round(one_alpha, 2))}")
        ylable_prefix = 'Average' if not best_codebook else 'Best Codebook'
        apply_log_ylable_prefix = '' if not best_codebook else 'Best Codebook '
        self._plot_computed_variables(wide=10,
                                      lines=self._plot_lines_with_appropiate_colors(number_modes, probs_labels),
                                      save_plot=save_plot if save_plot is not None else False,
                                      interactive_plot=interactive_plot if interactive_plot is not None else False,
                                      title=title,
                                      xlabel='number modes',
                                      ylabel=(f'{ylable_prefix} Success Probabilities'
                                              if not apply_log
                                              else f'{apply_log_ylable_prefix}Success Probability decreasing rate'),
                                      suffix=suffix,
                                      xtics=number_modes,
                                      specific_alphas=True)

    def success_probabilities(self,
                              number_modes: List[int],
                              number_ancillas: List[int],
                              global_results: List[GlobalResult],
                              save_plot: Optional[bool] = False,
                              interactive_plot: Optional[bool] = False,
                              best_codebook: Optional[bool] = False) -> None:
        probs_labels = []
        squeezing_options = [False, True]
        if best_codebook is None:
            best_codebook = False
        for number_mode in number_modes:
            helstrom_probabilities, homodyne_probabilities = self._compute_hels_homo_probs(
                alphas=self._alphas,
                apply_log=False,
                number_mode=number_mode,
                global_results=global_results,
                best_codebook=best_codebook)

            probs_labels.append((helstrom_probabilities, f'$pHel(a)^{number_mode}$'))
            for squeezing_option in squeezing_options:

                for number_ancilla in number_ancillas:
                    probs: List[float] = []
                    for alpha in self._alphas:
                        one_alpha_probs = [global_result.success_probability
                                           if not best_codebook else global_result.best_success_probability
                                           for global_result in global_results
                                           if (global_result.alpha == alpha and
                                               global_result.number_modes == number_mode and
                                               global_result.number_ancillas == number_ancilla and
                                               global_result.squeezing == squeezing_option)]
                        one_alpha_prob = (sum(one_alpha_probs) / len(one_alpha_probs)
                                          if len(one_alpha_probs) > 0 else 0.0)
                        probs.append(one_alpha_prob)

                    probs.extend([0.0] * (len(self._alphas) - len(probs)))
                    if len(probs) > len(self._alphas):
                        raise ValueError(f"len(probs): {len(probs)}")
                    one_prob_label = (probs, f"mode_{number_mode} squeez:{squeezing_option} anc:{number_ancilla}")
                    probs_labels.append(one_prob_label)
            probs_labels.append((homodyne_probabilities, f'$pHom(a)^{number_mode}$'))
        title_prefix = 'Average' if not best_codebook else 'Best Codebook'
        ylable_prefix = 'Average' if not best_codebook else 'Best Codebook'
        suffix_prefix = '' if not best_codebook else 'best_'
        self._plot_computed_variables(wide=17 if interactive_plot else 15,
                                      lines=self._set_plot_lines(probs_labels=probs_labels),
                                      save_plot=save_plot if save_plot is not None else False,
                                      interactive_plot=interactive_plot if interactive_plot is not None else False,
                                      title=f"{title_prefix} Success Probability Results",
                                      xlabel="alpha values",
                                      ylabel=f'{ylable_prefix} Success Probabilities',
                                      suffix=f"_{suffix_prefix}probs")

    def _plot_computed_variables(self,
                                 wide: int,
                                 lines: List[plt.Line2D],
                                 save_plot: bool,
                                 interactive_plot: bool,
                                 title: str,
                                 xlabel: str,
                                 ylabel: str,
                                 suffix=None,
                                 xtics: List[int] = None,
                                 specific_alphas: bool = False) -> None:
        wide = 17 if interactive_plot else 15
        fig, axes = plt.subplots(figsize=[wide, 8])
        plt.subplots_adjust(left=0.16, right=0.75)
        plt.title(title, fontsize=20)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        new_lines = lines.copy()
        plt_lines = []
        for line in new_lines:
            plt_line, = axes.plot(line[0], line[1], label=line[2], color=line[3], linestyle=line[4])
            plt_lines.append(plt_line)

        plt.legend(fancybox=True, bbox_to_anchor=(1.43, 1.01), loc='upper right',
                   ncol=2, facecolor='silver', framealpha=0.7, fontsize='small')
        fig.patch.set_facecolor('lightgrey')
        # fig.patch.set_alpha(0.1)
        axes.patch.set_facecolor('lightgrey')
        # axes.patch.set_alpha(0.1)
        if xtics is not None:
            axes.set_xticks(xtics)
        if interactive_plot:
            rax, labels = (self._set_interactive_labels_success_prob(plt_lines)
                           if not specific_alphas
                           else self._set_interactive_labels_specific_alphas(plt_lines))
            rax.set_facecolor('silver')
            # rax.patch.set_alpha(0.7)
            labels_activated = [False] * len(labels)
            check = CheckButtons(rax, labels, labels_activated)

            check.on_clicked(lambda x: self._interactive_plot(x, check, plt_lines, labels, specific_alphas))

        return self._show_or_save_plot(save_plot, suffix, fig)

    def _set_interactive_labels_success_prob(self, plt_lines) -> Tuple[plt.Axes, List[str]]:
        rax = plt.axes([0.0, 0.5, 0.12, 0.35])
        labels = ["mode_1", "mode_2", "mode_3", "mode_4", "mode_5", "mode_6",
                  "Ancillas: 0", "Ancillas: 1", "Ancillas: 2", "Ancillas: 3",
                  "Squeezing: True", "Squeezing: False"]

        [line.set_visible(False) for line in plt_lines if 'mode' in str(
            line.get_label()) or 'pHom' in str(line.get_label()) or 'pHel' in str(line.get_label())]

        return rax, labels

    def _set_interactive_labels_specific_alphas(self, plt_lines) -> Tuple[plt.Axes, List[str]]:
        rax = plt.axes([0.02, 0.7, 0.1, 0.2])
        labels = ["Ancillas: 0", "Ancillas: 1", "Ancillas: 2", "Ancillas: 3", "Squeezing: True", "Squeezing: False"]
        [line.set_visible(False) for line in plt_lines if 'Squeez' in str(line.get_label())]

        return rax, labels

    def _show_or_save_plot(self, save_plot: bool, suffix: str, fig: plt.Figure) -> None:
        if save_plot:
            fixed_path = _fix_path(path=self._path)
            fig.savefig(f'{fixed_path}plot_{set_current_time()}{suffix}.png')
            return
        plt.show()

    def _interactive_plot(self, label: str, input_check, input_plt_lines, input_labels, specific_alphas) -> None:

        label_checks = input_check.get_status()
        ancilla_0 = label_checks[-6]
        ancilla_1 = label_checks[-5]
        ancilla_2 = label_checks[-4]
        ancilla_3 = label_checks[-3]
        squeezing_true = label_checks[-2]
        squeezing_false = label_checks[-1]

        if specific_alphas:
            self._set_line_visibility_for_specific_alphas(input_plt_lines,
                                                          ancilla_0,
                                                          ancilla_1,
                                                          ancilla_2,
                                                          ancilla_3,
                                                          squeezing_true,
                                                          squeezing_false)
        if not specific_alphas:
            self._set_line_visibility_for_probs(label,
                                                input_plt_lines,
                                                input_labels,
                                                label_checks,
                                                ancilla_0,
                                                ancilla_1,
                                                ancilla_2,
                                                ancilla_3,
                                                squeezing_true,
                                                squeezing_false)

        plt.draw()

    def _set_line_visibility_for_specific_alphas(self,
                                                 input_plt_lines,
                                                 ancilla_0,
                                                 ancilla_1,
                                                 ancilla_2,
                                                 ancilla_3,
                                                 squeezing_true,
                                                 squeezing_false):
        for line in input_plt_lines:
            line_label = str(line.get_label())
            if 'Hom' not in line_label and 'Hel' not in line_label:
                line.set_visible(False)
            if (squeezing_true and 'pSucc Squeez' in line_label and ancilla_0 and 'anc:0' in line_label):
                line.set_visible(True)
            if (squeezing_true and 'pSucc Squeez' in line_label and ancilla_1 and 'anc:1' in line_label):
                line.set_visible(True)
            if (squeezing_true and 'pSucc Squeez' in line_label and ancilla_2 and 'anc:2' in line_label):
                line.set_visible(True)
            if (squeezing_true and 'pSucc Squeez' in line_label and ancilla_3 and 'anc:3' in line_label):
                line.set_visible(True)
            if (squeezing_false and 'pSucc No Squeez' in line_label and ancilla_0 and 'anc:0' in line_label):
                line.set_visible(True)
            if (squeezing_false and 'pSucc No Squeez' in line_label and ancilla_1 and 'anc:1' in line_label):
                line.set_visible(True)
            if (squeezing_false and 'pSucc No Squeez' in line_label and ancilla_2 and 'anc:2' in line_label):
                line.set_visible(True)
            if (squeezing_false and 'pSucc No Squeez' in line_label and ancilla_3 and 'anc:3' in line_label):
                line.set_visible(True)

    def _set_line_visibility_for_probs(self,
                                       label,
                                       input_plt_lines,
                                       input_labels,
                                       label_checks,
                                       ancilla_0,
                                       ancilla_1,
                                       ancilla_2,
                                       ancilla_3,
                                       squeezing_true,
                                       squeezing_false):
        if 'mode' in label:
            for line in input_plt_lines:
                line_label = str(line.get_label())
                if (squeezing_true and label in line_label and 'squeez:True' in line_label and
                        ancilla_0 and 'anc:0' in line_label):
                    line.set_visible(not line.get_visible())
                if (squeezing_true and label in line_label and 'squeez:True' in line_label and
                        ancilla_1 and 'anc:1' in line_label):
                    line.set_visible(not line.get_visible())
                if (squeezing_true and label in line_label and 'squeez:True' in line_label and
                        ancilla_2 and 'anc:2' in line_label):
                    line.set_visible(not line.get_visible())
                if (squeezing_true and label in line_label and 'squeez:True' in line_label and
                        ancilla_3 and 'anc:3' in line_label):
                    line.set_visible(not line.get_visible())
                if (squeezing_false and label in line_label and 'squeez:False' in line_label and
                        ancilla_0 and 'anc:0' in line_label):
                    line.set_visible(not line.get_visible())
                if (squeezing_false and label in line_label and 'squeez:False' in line_label and
                        ancilla_1 and 'anc:1' in line_label):
                    line.set_visible(not line.get_visible())
                if (squeezing_false and label in line_label and 'squeez:False' in line_label and
                        ancilla_2 and 'anc:2' in line_label):
                    line.set_visible(not line.get_visible())
                if (squeezing_false and label in line_label and 'squeez:False' in line_label and
                        ancilla_3 and 'anc:3' in line_label):
                    line.set_visible(not line.get_visible())
                self._ideal_prob_visibility(label, line, line_label, 'Hom')
                self._ideal_prob_visibility(label, line, line_label, 'Hel')
        if 'Squeezing' in label:
            squeez_text = 'squeez:True' if 'Squeezing: True' in label else 'squeez:False'
            for index, label_check in enumerate(label_checks):
                if label_check and index < 5:
                    mode_label = input_labels[index]
                    for line in input_plt_lines:
                        line_label = str(line.get_label())
                        if (mode_label in line_label and squeez_text in line_label and
                                ancilla_0 and 'anc:0' in line_label):
                            line.set_visible(not line.get_visible())
                        if (mode_label in line_label and squeez_text in line_label and
                                ancilla_1 and 'anc:1' in line_label):
                            line.set_visible(not line.get_visible())
                        if (mode_label in line_label and squeez_text in line_label and
                                ancilla_2 and 'anc:2' in line_label):
                            line.set_visible(not line.get_visible())
                        if (mode_label in line_label and squeez_text in line_label and
                                ancilla_3 and 'anc:3' in line_label):
                            line.set_visible(not line.get_visible())
        if 'Ancillas' in label:
            if 'Ancillas: 0' in label:
                ancilla_text = 'anc:0'
            if 'Ancillas: 1' in label:
                ancilla_text = 'anc:1'
            if 'Ancillas: 2' in label:
                ancilla_text = 'anc:2'
            if 'Ancillas: 3' in label:
                ancilla_text = 'anc:3'
            for index, label_check in enumerate(label_checks):
                if label_check and index < 5:
                    mode_label = input_labels[index]
                    for line in input_plt_lines:
                        line_label = str(line.get_label())
                        if (squeezing_false and mode_label in line_label and
                                ancilla_text in line_label and 'squeez:False' in line_label):
                            line.set_visible(not line.get_visible())
                        if (squeezing_true and mode_label in line_label and
                                ancilla_text in line_label and 'squeez:True' in line_label):
                            line.set_visible(not line.get_visible())

    def _ideal_prob_visibility(self, label, line, line_label, ideal_label):
        if '_1' in label and ideal_label in line_label and '1' in line_label:
            line.set_visible(not line.get_visible())
        if '_2' in label and ideal_label in line_label and '2' in line_label:
            line.set_visible(not line.get_visible())
        if '_3' in label and ideal_label in line_label and '3' in line_label:
            line.set_visible(not line.get_visible())
        if '_4' in label and ideal_label in line_label and '4' in line_label:
            line.set_visible(not line.get_visible())
        if '_5' in label and ideal_label in line_label and '5' in line_label:
            line.set_visible(not line.get_visible())
        if '_6' in label and ideal_label in line_label and '6' in line_label:
            line.set_visible(not line.get_visible())

    def distances(self,
                  number_modes: List[int],
                  number_ancillas: List[int],
                  global_results: List[GlobalResult],
                  save_plot: Optional[bool] = False,
                  interactive_plot: Optional[bool] = False) -> None:
        distances_labels = []
        squeezing_options = [False, True]

        for number_mode in number_modes:
            for squeezing_option in squeezing_options:
                for number_ancilla in number_ancillas:
                    distances = [global_result.distance_to_helstrom_probability
                                 for global_result in global_results
                                 if (global_result.number_modes == number_mode and
                                     global_result.number_ancillas == number_ancilla and
                                     global_result.squeezing == squeezing_option)]
                    one_distance_label = (
                        distances, f"mode_{number_mode} squeez:{squeezing_option} anc:{number_ancilla}")
                    distances_labels.append(one_distance_label)

        self._plot_computed_variables(wide=16 if interactive_plot else 15,
                                      lines=self._set_plot_lines(probs_labels=distances_labels),
                                      save_plot=save_plot if save_plot is not None else False,
                                      interactive_plot=interactive_plot if interactive_plot is not None else False,
                                      title="Distance to Helstrom Probability Results",
                                      xlabel="alpha values",
                                      ylabel='Distance to Helstrom Probability',
                                      suffix="_dist")

    def bit_error_rates(self,
                        number_modes: List[int],
                        number_ancillas: List[int],
                        global_results: List[GlobalResult],
                        save_plot: Optional[bool] = False,
                        interactive_plot: Optional[bool] = False) -> None:
        bit_error_labels = []
        squeezing_options = [False, True]

        for number_mode in number_modes:
            for squeezing_option in squeezing_options:
                for number_ancilla in number_ancillas:
                    bit_error_rates = [global_result.bit_error_rate
                                       for global_result in global_results
                                       if (global_result.number_modes == number_mode and
                                           global_result.number_ancillas == number_ancilla and
                                           global_result.squeezing == squeezing_option)]
                    one_bit_errorlabel = (
                        bit_error_rates, f"mode_{number_mode} squeez:{squeezing_option} anc:{number_ancilla}")
                    bit_error_labels.append(one_bit_errorlabel)

        self._plot_computed_variables(wide=16 if interactive_plot else 15,
                                      lines=self._set_plot_lines(probs_labels=bit_error_labels),
                                      save_plot=save_plot if save_plot is not None else False,
                                      interactive_plot=interactive_plot if interactive_plot is not None else False,
                                      title="Bit Error Rates Results",
                                      xlabel='alpha values',
                                      ylabel='Bit Error Rates',
                                      suffix="_bits")

    def times(self,
              number_modes: List[int],
              number_ancillas: List[int],
              global_results: List[GlobalResult],
              save_plot: Optional[bool] = False,
              interactive_plot: Optional[bool] = False) -> None:
        times_labels = []
        squeezing_options = [False, True]

        for number_mode in number_modes:
            for squeezing_option in squeezing_options:
                for number_ancilla in number_ancillas:
                    times = [global_result.time_in_seconds
                             for global_result in global_results
                             if (global_result.number_modes == number_mode and
                                 global_result.number_ancillas == number_ancilla and
                                 global_result.squeezing == squeezing_option)]
                    one_time_label = (times, f"mode_{number_mode} squeez:{squeezing_option} anc:{number_ancilla}")
                    times_labels.append(one_time_label)

        self._plot_computed_variables(wide=16 if interactive_plot else 15,
                                      lines=self._set_plot_lines(probs_labels=times_labels),
                                      save_plot=save_plot if save_plot is not None else False,
                                      interactive_plot=interactive_plot if interactive_plot is not None else False,
                                      title="Computation Time Results",
                                      xlabel='alpha values',
                                      ylabel='Computation Time (seconds)',
                                      suffix="_times")

    def plot_success_probabilities(self,
                                   executions: Optional[List[ResultExecution]] = None,
                                   save_plot: Optional[bool] = False,
                                   interactive_plot: Optional[bool] = False) -> None:

        executions_probs_labels = []
        # executions_probs_labels = [self._ideal_probabilities.p_ken_op]

        if executions is not None:
            executions_probs_labels += [(execution['p_helstrom'], f"$pHel(a)^{execution['number_modes'][0]}$")
                                        for execution in executions]
            executions_probs_labels += [(execution['p_homodyne'], f"$pHom(a)^{execution['number_modes'][0]}$")
                                        for execution in executions]
            executions_probs_labels += [(execution['p_succ'], execution['plot_label'])
                                        for execution in executions]
            self._plot_title(execution=executions[0])

        self._plot_computed_variables(wide=16 if interactive_plot else 15,
                                      lines=self._set_plot_lines(probs_labels=executions_probs_labels),
                                      save_plot=save_plot if save_plot is not None else False,
                                      interactive_plot=interactive_plot if interactive_plot is not None else False,
                                      title="Average Success Probabilities",
                                      xlabel='alpha values',
                                      ylabel='Average Success Probabilities',
                                      suffix="")

    def _plot_title(self, execution: ResultExecution) -> None:
        total_time = f"\n Total time: {set_friendly_time(execution['total_time']) if 'total_time' in execution else ''}"
        alpha_time = set_friendly_time(execution['total_time'] /
                                       len(execution['alphas'])) if 'total_time' in execution else ''
        total_time_per_alpha = f"\n Average one alpha computation time: {alpha_time}"
        plt.title(f"{execution['plot_title']}{total_time}{total_time_per_alpha}", fontsize=8)

    def _set_plot_lines(self,
                        probs_labels: List[Tuple[List[float], str]]) -> List[List[Any]]:

        lines = []
        for prob, label in probs_labels:
            prob.extend([0.0] * (len(self._alphas) - len(prob)))
            if label.find('pHom') != -1:
                color = 'black'
                linestyle = self._ideal_probs_style_lines(label)
            if label.find('pHel') != -1:
                color = 'dimgrey'
                linestyle = self._ideal_probs_style_lines(label)
            if label.find('pKenOp') != -1:
                color = 'silver'
                linestyle = 'dashdot'
            if label.find('pTF') != -1:
                color = 'red'
                linestyle = 'solid'
            if label.find('pFock') != -1:
                color = 'orange'
                linestyle = 'solid'
            if label.find('pGaus') != -1:
                color = 'blue'
                linestyle = 'solid'
            if label.find('squeez:False') != -1:
                linestyle = 'dashed'
            if label.find('squeez:True') != -1:
                linestyle = 'solid'
            if label.find('mode_1') != -1:
                if label.find('anc:0') != -1:
                    color = 'red'
                if label.find('anc:1') != -1:
                    color = 'tomato'
                if label.find('anc:2') != -1:
                    color = 'coral'
                if label.find('anc:3') != -1:
                    color = 'lightcoral'
            if label.find('mode_2') != -1:
                if label.find('anc:0') != -1:
                    color = 'saddlebrown'
                if label.find('anc:1') != -1:
                    color = 'chocolate'
                if label.find('anc:2') != -1:
                    color = 'sandybrown'
                if label.find('anc:3') != -1:
                    color = 'peachpuff'
            if label.find('mode_3') != -1:
                if label.find('anc:0') != -1:
                    color = 'darkorange'
                if label.find('anc:1') != -1:
                    color = 'orange'
                if label.find('anc:2') != -1:
                    color = 'bisque'
                if label.find('anc:3') != -1:
                    color = 'papayawhip'
            if label.find('mode_4') != -1:
                if label.find('anc:0') != -1:
                    color = 'gold'
                if label.find('anc:1') != -1:
                    color = 'yellow'
                if label.find('anc:2') != -1:
                    color = 'khaki'
                if label.find('anc:3') != -1:
                    color = 'lemonchiffon'
            if label.find('mode_5') != -1:
                if label.find('anc:0') != -1:
                    color = 'olivedrab'
                if label.find('anc:1') != -1:
                    color = 'yellowgreen'
                if label.find('anc:2') != -1:
                    color = 'greenyellow'
                if label.find('anc:3') != -1:
                    color = 'lightgreen'
            if label.find('mode_6') != -1:
                if label.find('anc:0') != -1:
                    color = 'teal'
                if label.find('anc:1') != -1:
                    color = 'c'
                if label.find('anc:2') != -1:
                    color = 'cyan'
                if label.find('anc:3') != -1:
                    color = 'lightcyan'
            if label.find('mode_7') != -1:
                if label.find('anc:0') != -1:
                    color = 'steelblue'
                if label.find('anc:1') != -1:
                    color = 'deepskyblue'
                if label.find('anc:2') != -1:
                    color = 'skyblue'
                if label.find('anc:3') != -1:
                    color = 'lightskyblue'
            if label.find('mode_8') != -1:
                if label.find('anc:0') != -1:
                    color = 'navy'
                if label.find('anc:1') != -1:
                    color = 'blue'
                if label.find('anc:2') != -1:
                    color = 'cornflowerblue'
                if label.find('anc:3') != -1:
                    color = 'lavender'
            if label.find('mode_9') != -1:
                if label.find('anc:0') != -1:
                    color = 'indigo'
                if label.find('anc:1') != -1:
                    color = 'darkviolet'
                if label.find('anc:2') != -1:
                    color = 'magenta'
                if label.find('anc:3') != -1:
                    color = 'plum'
            if label.find('mode_10') != -1:
                if label.find('anc:0') != -1:
                    color = 'crimson'
                if label.find('anc:1') != -1:
                    color = 'deeppink'
                if label.find('anc:2') != -1:
                    color = 'paleovioletred'
                if label.find('anc:3') != -1:
                    color = 'lightpink'
            line = [self._alphas, prob, label, color, linestyle]
            lines.append(line)
        return lines

    def _ideal_probs_style_lines(self, label) -> str:
        if label.find('1') != -1:
            linestyle = 'solid'
        if label.find('2') != -1:
            linestyle = 'dashdot'
        if label.find('3') != -1:
            linestyle = 'dotted'
        if label.find('4') != -1:
            linestyle = 'solid'
        if label.find('5') != -1:
            linestyle = 'dashdot'
        if label.find('6') != -1:
            linestyle = 'dotted'
        if label.find('7') != -1:
            linestyle = 'solid'
        if label.find('8') != -1:
            linestyle = 'dashdot'
        if label.find('9') != -1:
            linestyle = 'dotted'
        if label.find('10') != -1:
            linestyle = 'solid'
        if label.find('11') != -1:
            linestyle = 'dashdot'
        if label.find('12') != -1:
            linestyle = 'dotted'
        return linestyle
