import pygame
import sys
from pygame import *
from random import randint
MOVE_SLEEP = 0.01
import time
class TankMain():
    width = 600
    height = 500
    desTime = 0
    grade = 0
    myShells = []
    enemyList = pygame.sprite.Group()
    def setLife(self,live=3):
        self.life = int(live)
    def getGrade(self):
        text = pygame.font.Font("./font/msyhbd.ttc",20).render("分数：{} 生命值：{}".format(self.grade,self.life),True,(0,255,0))
        return text
    def startGame(self):
        pygame.init()#加载操作系统资源
        self.screen = pygame.display.set_mode((self.width,self.height),0,32)
        pygame.display.set_caption("打唐克")
        self.myTank = MyTank(self.screen)
        for i in range(10):
            self.enemyList.add(BnemyTank(self.screen))
        #更新窗口
        while True:
            if len(self.enemyList)<6:
                if time.time() - self.desTime >1:
                    self.enemyList.add(BnemyTank(self.screen))
                self.screen.fill((0,0,0))
                #监听键盘
            key = pygame.key.get_pressed()
            if key[K_LEFT]:
                self.myTank.move('L')
                print('1')
            elif key[K_RIGHT] :
                self.myTank.move('R')
            elif key[K_UP]:
                self.myTank.move('U')
            elif key[K_DOWN]:
                self.myTank.move('D')
            else:
                pass
            self.get_event()
            [enemy.moveMore() for enemy in self.enemyList]
            self.myTank.display()
            [enemy.display() for enemy in self.enemyList]
            [shell.move() for shell in self.myShells]
            for shell in self.myShells:
                b = shell.move()
                if b ==True:
                    self.myShells.remove(shell)
                    b = 0
                a = shell.hitTank()
                if a == True:
                    self.myShells.remove(shell)
                    self.grade += 1
                    self.desTime = time.time()
                    a = 0
            if self.myTank.live == True:
                a = self.myTank.hitTank()
                if a == True:
                    self.life -= 1
                    if self.life <= 0:
                        self.myTank.live = False
                    else:
                        self.myTank = MyTank(self.screen)
            [shell.display() for shell in self.myShells]
            self.screen.blit(self.getGrade(),(5,5))
            if self.myTank.live == False:
                self.stopGamePrint()
            pygame.display.update()
            time.sleep(MOVE_SLEEP)
    def get_event(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.stopGame()
            if event.type == KEYDOWN:
                if event.key == K_SPACE:
                    self.myShells.append(self.myTank.fire())
                if event.key == K_ESCAPE:
                    self.stopGame()

    def stopGame(self):
        sys.exit()
    def stopGamePrint(self):
        text = pygame.font.Font('./font/msyh.ttc',70).render("game over !",True,(0,255,0))
        self.screen.blit(text,(100,200))






#所有坦克的基础类
class BaseItem(pygame.sprite.Sprite):
    def __init__(self,screen):
        self.screen = screen
        pygame.sprite.Sprite.__init__(self)
#坦克类
class Tank(BaseItem):
    width = 50
    height = 50
    time = 0
    direction = 'U'
    images = {}
    def __init__(self,screen,left,top):
        super().__init__(screen)
        self.screen = screen
        self.speed = 2
        self.images['L'] = pygame.image.load('./images/04.jpg')
        self.images['R'] = pygame.image.load('./images/02.jpg')
        self.images['U'] = pygame.image.load('./images/01.jpg')
        self.images['D'] = pygame.image.load('./images/03.jpg')
        self.image = self.images[self.direction]
        self.rect = self.image.get_rect()
        self.rect.left = left
        self.rect.top = top
        self.live = True #坦克是否被消灭  ？？？？

    def display(self):
        self.time += 1
        if self.live == True:
            self.image = self.images[self.direction]
            self.screen.blit(self.image,self.rect)
    def isObstacle(self):
        tag = ''
        if self.rect.left <=0: tag += 'L'
        if self.rect.left >= TankMain.width - self.width: tag += 'R'
        if self.rect.top <=0: tag += 'U'
        if self.rect.top >= TankMain.height - self.height: tag += 'D'
        return tag
    def fire(self):
        m = Shell(self.screen,self)
        return m
    def move(self,direction):
        if self.live == True:
            if self.time > 1 / MOVE_SLEEP:
                if direction == self.direction:
                    obstacle = self.isObstacle()
                    if self.direction == 'L' and ("L" not in obstacle):
                        self.rect.left -= self.speed
                    elif self.direction == 'R' and ('R' not in obstacle):
                        self.rect.left += self.speed
                    elif self.direction == 'U' and ("U" not in obstacle):
                        self.rect.top -= self.speed
                    elif self.direction == 'D' and ('D' not in obstacle):
                        self.rect.top += self.speed
                    else:
                        pass
                else:
                    self.direction = direction

class MyTank(Tank):
    images = {}
    live = True
    def __init__(self,screen):
        super().__init__(screen,275,450)
        self.images['L'] = pygame.image.load("images/4.jpg")
        self.images['R'] = pygame.image.load("images/2.jpg")
        self.images['U'] = pygame.image.load("images/1.jpg")
        self.images['D'] = pygame.image.load("images/3.jpg")
        self.image = self.images[Tank.direction]
        self.screen = screen
        self.rect = self.image.get_rect()
        self.rect.left = 275
        self.rect.top = 450
    def display(self):
        self.time+= 1
        self.image = self.images[self.direction]
        self.screen.blit(self.image,self.rect)
    def hitTank(self):
        hitList = pygame.sprite.spritecollide(self,TankMain.enemyList,False)
        for e in hitList:
            self.live = False
            return True
        return False
class BnemyTank(Tank):
    def __init__(self,screen):
        super().__init__(screen,randint(1,5)*100,0)
        self.getDirection()
        self.step = 0
        self.speed = 1
    def getDirection(self):
        self.direction = ['L','R','U','D'][randint(0,3)]
    def moveMore(self):
        if self.live == True:
            if self.step == 0 or (self.direction in self.isObstacle()):
                self.getDirection()
                self.step = randint(0,200)
            else:
                self.move(self.direction)
                self.step -= 1
class Shell(BaseItem):
    width = 12
    height = 12
    def __init__(self,screen,tank):
        super().__init__(screen)
        self.image = pygame.image.load('./images/3.png')
        self.direction = tank.direction
        self.rect = self.image.get_rect()
        self.rect.left = tank.rect.left + (tank.width - self.width)/2.0
        self.rect.top = tank.rect.top + (tank.height - self.height) /2.0
        self.speed = 3
        self.live = True
    def isObstacle(self):
        tag = ''
        if self.rect.left <= 0: tag += 'L'
        if self.rect.left >= TankMain.width - self.width: tag += 'R'
        if self.rect.top <= 0: tag += 'U'
        if self.rect.top >= TankMain.height - self.height: tag += 'D'
        return tag
    def move(self):
        if self.live == True:
            obstacle = self.isObstacle()
            if self.direction == 'L' and ('L' not in obstacle):
                self.rect.left -= self.speed
            elif self.direction == 'R' and ('R' not in obstacle):
                self.rect.left += self.speed
            elif self.direction == 'U' and ('U' not in obstacle):
                self.rect.top -= self  .speed
            elif self.direction == 'D' and ('D' not in obstacle):
                self.rect.top += self.speed
            else:
                return True
        else:
            pass
    def display(self):
        if self.live == True or self.live == False:
            self.screen.blit(self.image,self.rect)
    def hitTank(self):
        hitList = pygame.sprite.spritecollide(self,TankMain.enemyList,False)
        for e in hitList:
            e.live = False
            TankMain.enemyList.remove(e)
            self.live = False
            return True
        return False
class Blast(BaseItem):
    def __init__(self,screen,rect):
        super().__init__(screen)
        self.rect = rect


if __name__ == '__main__':
    game = TankMain()
    game.setLife()
    game.startGame()