import tensorflow as tf
import scipy.misc as misc
import scipy.io
import numpy as np
import PIL.Image as Image

style_img_path = "./input/style.png"
content_img_path = "./input/content.png"
result_img_path = "./output/result.jpg"
img_width = 512
img_height = 512
img_channels = 3
iterations = 100

# content loss weight
alpha = 1e-5
# style loss weight
beta = 1.0

vgg19_net_path = "./vgg.mat"


# def mapping(img):
#     return 255.0 * (img - np.min(img)) / (np.max(img) - np.min(img))


def loadImg(img_path):
    return np.reshape(misc.imresize(np.array(Image.open(img_path)), [img_height, img_width], mode="RGB"),
                      [1, img_height, img_width, img_channels])


def loadVGG19(vgg_path):
    return scipy.io.loadmat(vgg_path)


def getFeature(input, vgg):
    layers = vgg["layers"]
    feature = {}
    with tf.variable_scope("vgg"):
        for i in range(37):
            if layers[0, i][0, 0]["type"] == "conv":
                w = layers[0, i][0, 0]["weights"][0, 0]
                b = layers[0, i][0, 0]["weights"][0, 1]
                with tf.variable_scope(str(i)):
                    w = tf.constant(w)
                    b = tf.constant(b)
                    input = tf.nn.conv2d(input, w, [1, 1, 1, 1], "SAME") + b
            elif layers[0, i][0, 0]["type"] == "relu":
                input = tf.nn.relu(input)
                feature[layers[0, i][0, 0]["name"][0]] = input
            elif layers[0, i][0, 0]["type"] == "pool":
                input = tf.nn.avg_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
            else:
                pass

    return feature


def saveImg(img):
    mapping = 255.0 * (img - np.min(img)) / (np.max(img) - np.min(img))

    # 255.0 * (img - np.min(img)) / (np.max(img) - np.min(img))

    Image.fromarray(np.uint8(mapping(np.reshape(img, [img_height, img_width, img_channels])))).save(
        result_img_path)

    # Image.fromarray(np.uint8(mapping(np.reshape(target_img, [H, W, C])))).save("./deepdream/target_pre.jpg")


def getContentLoss(feature_content, feature_result):
    return tf.reduce_sum(tf.square(feature_result["relu4_2"] - feature_content["relu4_2"])) * 0.5


def getStyleLoss(feature_style, feature_result):
    E = 0
    for layer in feature_style.keys():
        w = 0.0
        if layer == "relu1_1" or layer == "relu2_1" or layer == "relu3_1":
            w = 0.3
        else:
            w = 0.0
        C = int(feature_result[layer].shape[-1])
        H = int(feature_result[layer].shape[1])
        W = int(feature_result[layer].shape[2])
        F = tf.reshape(tf.transpose(feature_result[layer], [0, 3, 1, 2]), shape=[C, -1])
        # Gram matrix of result(x)
        G_x = tf.matmul(F, tf.transpose(F))

        C = int(feature_style[layer].shape[-1])
        F = tf.reshape(tf.transpose(feature_style[layer], [0, 3, 1, 2]), shape=[C, -1])
        # Gram matrix of style
        G_s = tf.matmul(F, tf.transpose(F))
        E += w * tf.reduce_sum(tf.square(G_x - G_s)) / (4 * C ** 2 * H ** 2 * W ** 2)
    return E


def styleTransfer(sess):
    content = tf.placeholder("float", [1, img_height, img_width, img_channels])
    style = tf.placeholder("float", [1, img_height, img_width, img_channels])
    result = tf.get_variable("result", shape=[1, img_height, img_width, img_channels],
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
    vgg = loadVGG19(vgg19_net_path)

    result_feature = getFeature(result, vgg)
    style_feature = getFeature(style, vgg)
    content_feature = getFeature(content, vgg)

    style_loss = getStyleLoss(style_feature, result_feature)
    content_loss = getContentLoss(content_feature, result_feature)
    total_loss = alpha * content_loss + beta * style_loss

    optimizer = tf.contrib.opt.ScipyOptimizerInterface(total_loss, method='L-BFGS-B',
                                                       options={'maxiter': iterations, 'disp': 0})

    sess.run(tf.global_variables_initializer())

    content_img = loadImg(content_img_path)
    style_img = loadImg(style_img_path)
    sess.run(tf.compat.v1.assign(result, content_img), feed_dict={content: content_img, style: style_img})

    optimizer.minimize(sess, feed_dict={content: content_img, style: style_img})
    #
    # content_loss = sess.run(content_loss, feed_dict={content: content_img, self.style_img: style_img})
    # style_loss = sess.run(style_loss, feed_dict={self.content_img: content_img, self.style_img: style_img})
    # total_loss = sess.run(total_loss, feed_dict={self.content_img: content_img, self.style_img: style_img})
    content_loss = sess.run(content_loss, feed_dict={content: content_img, style: style_img})
    style_loss = sess.run(style_loss, feed_dict={content: content_img, style: style_img})
    total_loss = sess.run(total_loss, feed_dict={content: content_img, style: style_img})

    print("l loss")
    print("L_content: %g, L_style: %g, L_total: %g" % (content_loss, style_loss, total_loss))

    result = sess.run(result, feed_dict={content: content_img, style: style_img})

    def mapping(img):
        255.0 * (img - np.min(img)) / (np.max(img) - np.min(img))

    Image.fromarray(np.uint8(mapping(np.reshape(result, [img_height, img_width, img_channels])))).save(
        result_img_path)


if __name__ == '__main__':
    with tf.compat.v1.Session() as sess:
        # load content image, load style image, generate white noise image and load vgg19
        styleTransfer(sess)
        # saveImg(result_img)
