//
// Created by jianfei on 16-10-19.
//

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "Tree.h"
#include "Document.h"

TEST(Tree, pos) {
    Tree tree(3, 0.5);

    auto *root = tree.GetRoot();
    auto *node1 = tree.AddChildren(root);
    auto *node2 = tree.AddChildren(node1);

    auto *node3 = tree.AddChildren(node1);

    Document doc1, doc2;
    doc1.c = decltype(doc1.c){root, node1, node2};
    doc2.c = decltype(doc2.c){root, node1, node3};

    tree.UpdateNumDocs(doc1.c.back(), 1);
    tree.UpdateNumDocs(doc2.c.back(), 1);

    // Layer size
    EXPECT_EQ(tree.NumNodes(0), 1);
    EXPECT_EQ(tree.NumNodes(1), 1);
    EXPECT_EQ(tree.NumNodes(2), 2);

    // ID
    EXPECT_EQ(root->pos, 0);
    EXPECT_EQ(node1->pos, 0);
    EXPECT_EQ(node2->pos, 0);
    EXPECT_EQ(node3->pos, 1);

    // DocID
    ASSERT_THAT(doc1.GetPos(), testing::ElementsAre(0, 0, 0));
    ASSERT_THAT(doc2.GetPos(), testing::ElementsAre(0, 0, 1));

    tree.UpdateNumDocs(doc1.c.back(), -1);
    auto result = tree.Compress(2);

    ASSERT_THAT(result, testing::ElementsAre(-1, 0));

    EXPECT_EQ(root->pos, 0);
    EXPECT_EQ(node1->pos, 0);
    EXPECT_EQ(node3->pos, 0);

    ASSERT_THAT(doc2.GetPos(), testing::ElementsAre(0, 0, 0));
}
