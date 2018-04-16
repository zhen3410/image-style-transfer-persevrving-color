import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import os
import sys
import argparse
import model
import pdb

INI_NOISE_RATIO = 0.7
STYLE_STRENGTH = 500
ITERATION=2000
HIGHT=600
WIDTH=800


def build_content_loss(p, x):
    M = p.shape[1] * p.shape[2]
    N = p.shape[3]
    loss = (1./(2* N**0.5 * M**0.5 )) * tf.reduce_sum(tf.square(tf.subtract(x , p)))
    return loss


def histogram_match(content, style):
    H, W, _ = content.shape
    style = scipy.misc.imresize(style, (H, W))

    cont_mu = np.mean(content, axis=(0, 1))
    sty_mu = np.mean(style, axis=(0, 1))

    cont_1 = content - cont_mu.reshape((1, 1, 3))
    sty_1 = style - sty_mu.reshape((1, 1, 3))
    cont_cov = np.dot(np.concatenate(cont_1).T,
                      np.concatenate(cont_1)) / (H * W)
    sty_cov = np.dot(np.concatenate(sty_1).T, np.concatenate(sty_1)) / (H * W)

    style_val, style_vec = np.linalg.eig(sty_cov)
    style_val_1 = np.diag(np.sqrt(style_val))
    sty_sigma_1 = style_vec.dot(style_val_1).dot(style_vec.T)
    sty_sigma_inv = np.linalg.inv(sty_sigma_1)

    cont_sty_cov = sty_sigma_1.dot(cont_cov).dot(sty_sigma_1)
    contant_val, contant_vec = np.linalg.eig(cont_sty_cov)
    contant_val_1 = np.diag(np.sqrt(contant_val))
    cont_sigma_1 = contant_vec.dot(contant_val_1).dot(contant_vec.T)
    A_IA = sty_sigma_inv.dot(cont_sigma_1).dot(sty_sigma_inv)
    b = cont_mu - A_IA.dot(sty_mu.T)

    style_pram = np.add(np.dot(style, A_IA.T), b)

    return style_pram


def rgb2yiq(rgb):
    yiq_mat = np.array(
        [[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
    rgb_yiq = np.dot(rgb, yiq_mat.T).astype(np.float32)
    return rgb_yiq


def yiq2rgb(yiq):
    rgb_mat = np.array(
        [[1.000, 0.956, 0.621], [1.000, -0.273, -0.647], [1.000, -1.104, 1.701]])
    yiq_rgb = np.dot(yiq, rgb_mat.T).astype(np.float32)
    return yiq_rgb


def luminance_transfer(content, style):
    # pdb.set_trace()
    content_yiq = rgb2yiq(content)
    style_yiq = rgb2yiq(style)

    content_yiq_lum = np.zeros_like(content_yiq)
    content_yiq_col = np.zeros_like(content_yiq)
    style_yiq_lum = np.zeros_like(style_yiq)
    # style_yiq_col = np.zeros_like(style_yiq)

    con_mu = np.mean(content_yiq[:, :, 0])
    sty_mu = np.mean(style_yiq[:, :, 0])
    con_std = np.std(content_yiq[:, :, 0])
    sty_std = np.std(style_yiq[:, :, 0])
    style_yiq_lum[:, :, 0] = np.add(
        np.dot((con_std / sty_std), (style_yiq[:, :, 0] - sty_mu)), con_mu)
    # style_yiq_col[:,:,1:2]=style_yiq[:,:,1:2]
    content_yiq_lum[:, :, 0] = content_yiq[:, :, 0]
    content_yiq_col[:, :, 1:3] = content_yiq[:, :, 1:3]

    style_yiq_lum = yiq2rgb(style_yiq_lum)
    content_yiq_lum = yiq2rgb(content_yiq_lum)

    return content_yiq_lum, style_yiq_lum, content_yiq_col


def gram_matrix(x, area, depth):
    x1 = tf.reshape(x, (area, depth))
    g = tf.matmul(tf.transpose(x1), x1)
    return g

def build_style_loss(a, x):
    M = a.shape[1] * a.shape[2]
    N = a.shape[3]
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1. / (4 * N**2 * M**2)) * \
            tf.reduce_sum(tf.square(tf.subtract(G, A)))
    return loss


def read_image(path):
    image = scipy.misc.imread(path)
    image=scipy.misc.imresize(image,(HIGHT,WIDTH))
    image = image[np.newaxis, :, :, :]
    image = image - model.MEANS
    return image


def write_image(path, image):
    image = image + model.MEANS
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)


def write_image_lum(path, image, content_yiq_col):
    image = image + model.MEANS
    image = rgb2yiq(image[0])
    content_yiq_col[:, :, 0] = image[:, :, 0]
    content = yiq2rgb(content_yiq_col)
    scipy.misc.imsave(path, content)


def main(args):
    content_img = read_image(args.content)
    _,H,W,_ = content_img.shape
    style_img=read_image(args.style)
    #pdb.set_trace()
    net = model.load_vgg_model(model.VGG_MODEL, H, W)
    noise_img = np.random.uniform(-20, 20, (1, H, W, 3)).astype('float32')

    if args.preserve_color_method == 'histogram_match':
        style_img[0] = histogram_match(content_img[0], style_img[0])
    elif args.preserve_color_method == 'luminance':
        content_img[0], style_img[0], content_yiq_col = luminance_transfer(content_img[0], style_img[0])

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    sess.run([net['input'].assign(content_img)])
    cost_content = sum(map(lambda l, : l[1] * build_content_loss(sess.run(net[l[0]]),  net[l[0]]), model.CONTENT_LAYERS))

    sess.run([net['input'].assign(style_img)])
    cost_style = sum(map(lambda l: l[1] * build_style_loss(sess.run(net[l[0]]),  net[l[0]]), model.STYLE_LAYERS))

    cost_total = cost_content + STYLE_STRENGTH * cost_style
    optimizer = tf.train.AdamOptimizer(2.0)

    train_step = optimizer.minimize(cost_total)
    sess.run(tf.initialize_all_variables())
    sess.run(net['input'].assign( INI_NOISE_RATIO* noise_img + (1.-INI_NOISE_RATIO) * content_img))
    
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    for i in range(ITERATION):
        sess.run(train_step)
        if i%100==0:
            result_img = sess.run(net['input'])
            Jt=sess.run([cost_total])
            print("Iteration"+str(i)+":")
            print("total cost ="+str(Jt))
            #pdb.set_trace()
            if args.preserve_color_method=='luminance':
                write_image_lum(os.path.join(args.output, '%s.jpg' %(str(i).zfill(4))), result_img,content_yiq_col)
            else:
                write_image(os.path.join(args.output, '%s.jpg' %(str(i).zfill(4))),result_img)

    if args.preserve_color_method=='luminance':
        write_image_lum(os.path.join(args.output, 'results.jpg'), result_img,content_yiq_col)
    else:
        write_image(os.path.join(args.output, 'results.jpg'), result_img)


def parser_argument(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--content', dest='content',
                        default='./images/in3.png')
    parser.add_argument('--style', dest='style', default='./images/trial.jpg')
    parser.add_argument('--output', dest='output', default='./output2')
    parser.add_argument('--p',dest='preserve_color_method', default='histogram_match')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parser_argument(sys.argv[1:]))
