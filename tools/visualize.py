import dominate
from dominate.tags import *
import os
import numpy as np
import ntpath
import time
import glob


# define HTML class
class HTML:
    def __init__(self, title, reflesh=0):
        self.title = title

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))


    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, width=400):
        self.add_table()
        with self.t:
            with tr():
                for im, txt in zip(ims, txts):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=im):
                                img(style="width:%dpx" % width, src=im)
                            br()
                            p(txt)

    def save(self):
        html_file = '%s.html' % self.title
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


# Visualizer class
class Visualizer():
    def __init__(self, name):
        self.win_size = 256
        self.name = name

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch):
        if self.display_id > 0: # show images in the browser
            if self.display_single_pane_ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
    table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
    table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
</style>""" % (w, h)
                ncols = self.display_single_pane_ncols
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win = self.display_id + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    #image_numpy = np.flipud(image_numpy)
                    self.vis.image(image_numpy.transpose([2,0,1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                    idx += 1

        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)


def main():

    web_name = 'Baseline_comparison'
    suffix = ['', '_dcp_radiance-refinedt', '_DehazeNet', '_Haze-free']
    folder = ['IVCDehazingDataset/test/testA', 
              'IVCDehazingDataset/IVC_DarkChannelPrior', 
              'IVCDehazingDataset/IVC_DehazeNet', 
              'IVCDehazingDataset/recongan_nyu/test_latest/images']

    assert len(suffix) == len(folder)

    img_names = glob.glob(folder[0]+'/*.png')
    max_num = 100
    win_size = 256

    # create website
    webpage = HTML(web_name)

    for i, name in enumerate(img_names):
        if i >= max_num:
            break

        name = os.path.basename(name)
        img_path = []
        for j in range(len(suffix)):
            img_path.append(os.path.join(folder[j], name.replace('.png', suffix[j]+'.png')))

        webpage.add_header(name)
        webpage.add_images(img_path, suffix, win_size)

    webpage.save()

if __name__ == '__main__':
    main()
